"""
This script is a template/inspiration that you can use to build the models for benchmark purpose.
It will output a csv file with the statistics called build_qubos.csv in the current directory.

To run this script:

- create benchmark datasets: the convention is a directory per density (-n option of create_dataset),
  round to an integer and with the ds prefix: density 0.1 in folder ds10, etc.
- update the values in BUILD CONFIG

"""

from hepqpr.qallse.cli.func import *

# ==== BUILD CONFIG TODO change it

loglevel = logging.DEBUG

events = [1000 + x for x in range(10)] # events to run
dss = [100] # densities, here just ds100

data_path = 'C:/Users/timsc/hepqpr-qallse-data/evt{event}/ds{ds}/event00000{event}-hits.csv'

output_path = 'C:/Users/timsc/hepqpr-qallse-data/evt{event}/ds{ds}/'
output_prefix = 'evt{event}-ds{ds}-'

model_class = QallseD0  # model class to use
extra_config = dict()  # model config

dump_config = dict(
    xplets_kwargs=dict(),  # use json or "pickle"
    qubo_kwargs=dict(w_marker=None, c_marker=None)  # save the real coefficients VS generic placeholders
)

# ==== configure logging

init_logging(logging.DEBUG, sys.stdout)

# ==== build model

if __name__ == '__main__':
    mat = []

    for event in events:
        for ds in dss:
            print(f'\n==========>>> processing event {event} ds{ds}\n', flush=True)

            # load data
            path = data_path.format(event=event, ds=ds)
            dw = DataWrapper.from_path(path)
            doublets = pd.read_csv(path.replace('-hits.csv', '-doublets.csv'))

            # build model
            with time_this() as time_info:
                model = model_class(dw, **extra_config)
                model.build_model(doublets)

            # dump model to a file
            if not os.path.exists(output_path.format(event=event, ds=ds)):
                    os.makedirs(output_path.format(event=event, ds=ds))

            Q = dumper.dump_model(
                model,
                output_path=output_path.format(event=event, ds=ds),
                prefix=output_prefix.format(event=event, ds=ds),
                **dump_config
            )

            # gather stats
            mat.append(
                [
                    event, ds,
                    len(model.qubo_doublets),
                    len(model.qubo_triplets),
                    len(model.quadruplets),
                    len(Q)
                ] + time_info)

        headers = 'event,percent,n_doublets,n_triplets,n_qplets,q,cpu_time,wall_time'
        stats = pd.DataFrame(mat, columns=headers.split(','))
        stats.to_csv('build_qubo.csv', index=False)
