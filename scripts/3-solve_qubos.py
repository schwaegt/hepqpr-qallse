"""
This script is a template/inspiration that you can use to solve qubos for benchmark purpose.
It will output a csv file with the statistics called solve_qubos_{solver}.csv in the current directory.

To run this script:

- create benchmark datasets: the convention is a directory per density (-n option of create_dataset),
  round to an integer and with the ds prefix: density 0.1 in folder ds10, etc.
- build qubos using for example the script 2-build_qubos.py
- update the values in BUILD CONFIG

"""
import pickle
import random

from hepqpr.qallse.cli.func import *
from hepqpr.qallse.cli.vqe import *
import os.path as op

# ==== RUN CONFIG TODO change it

loglevel = logging.DEBUG

events = [1000]
dss = [5]
repeat = 1

data_path = 'C:/Users/timsc/hepqpr-qallse-data/evt{evt}/ds{ds}/event00000{evt}-hits.csv'  # path to the datasets
qubo_path = 'C:/Users/timsc/hepqpr-qallse-data/evt{evt}/ds{ds}/'  # path where the qubos are pickled
qubo_slice_prefix = 'evt{event}-ds{ds}-sz{sz}-ov{ov}-slice{slice}'
output_path = 'C:/Users/timsc/hepqpr-qallse-data/evt{evt}/ds{ds}/'  # where to serialize the responses
stats_path = 'C:/Users/timsc/hepqpr-qallse-data/stats/'

solver_config = {
'solver': 'neal', # solver to use
'backend_name': 'ibmq_qasm_simulator',
'qubo_slice_size': 23,
'overlap': 5,
'sub_qubo_size': 19,
'su2_gates': 'ry',
'reps': 0,
'entanglement': 'linear',
'optimizer_name': 'NFT',
'maxiter': 1024,
'shots': 1024,
'sub_qubo_iterations': 10,
'vqe_repetitions': 10,
'reduce_qubo': False,
'measurement_error_mitigation': None,
} # parameters for the solver. Note that "seed" is generated later.
size = solver_config['qubo_slice_size']
overlap = solver_config['overlap']
solver = solver_config['solver']
# ==== configure logging

init_logging(logging.DEBUG, sys.stdout)


# ==== build model

def run_one(event, ds):
    # load data
    path = data_path.format(evt=event, ds=ds)
    dw = DataWrapper.from_path(path)
    qubo_filepath = op.join(qubo_path.format(evt=event, ds=ds), 'evt{evt}-ds{ds}-qubo.pickle'.format(evt=event, ds=ds))
    vars_predetermined = {}

    with open(qubo_filepath, 'rb') as f:
        Q = pickle.load(f)
    en0 = dw.compute_energy(Q)

    if solver_config['reduce_qubo']:
        Q, vars_predetermined = reduce_qubo(Q)

    xplet_filepath  = op.join(qubo_path.format(evt=event, ds=ds), 'evt{evt}-ds{ds}-xplets.pickle'.format(evt=event, ds=ds))
    with open(xplet_filepath, 'rb') as f:
        xplets = pickle.load(f)
    #slice qubo for VQE
    if solver == 'vqe_slices' or solver == 'eigensolver_slices' or solver == 'neal_slices':
        #list of QUBOS
        Q_slices = slice_qubo(Q, xplets, solver_config['qubo_slice_size'], solver_config['overlap'])
        print('Finished slicing')
        #save QUBO slices
        # for count, qubo_slice in enumerate(Q_slices):
        #     qubo_slice_path = op.join(qubo_path.format(ds=ds), 'qubo-slices-sz{sz}-ov{ov}/'.format(sz=size, ov=overlap), qubo_slice_prefix.format(event=event, ds=ds, sz=size, ov=overlap, slice=count))
        #     if not os.path.exists(qubo_slice_path):
        #         os.makedirs(qubo_slice_path)
        #     qubo_slice_filepath = op.join(qubo_slice_path, qubo_slice_prefix.format(event=event, ds=ds, sz=size, ov=overlap, slice=count) + '-qubo.pickle')
        #     with open(qubo_slice_filepath, 'wb') as f:
        #         pickle.dump(qubo_slice, f)

    for i in range(repeat):
        # set seed
        seed = random.randint(0, 1 << 30)
        #solver_config.update({'seed': seed})
        unique_prefix = str(event) + '-' + str(ds)
        for value in solver_config.values():
            unique_prefix += '-' + str(value)


        # build model
        with time_this() as time_info:
            with time_this() as qtime_info:
                if solver == 'neal':
                    response = solve_neal(Q)
                elif solver == 'neal_slices':
                    response = solve_neal_slices(Q_slices)
                elif solver == 'qbsolv':
                    response = solve_qbsolv(Q, **solver_config)
                elif solver == 'dwave':
                    response = solve_dwave(Q, **solver_config)
                elif solver == 'vqe_slices':
                    response = solve_vqe_slices(Q_slices, **solver_config)
                elif solver == 'vqe_sub_qubos':
                    response = solve_vqe_sub_qubos(Q, xplets, **solver_config)
                elif solver == 'eigensolver_slices':
                    response = solve_eigensolver_slices(Q_slices, **solver_config)
                elif solver == 'eigensolver_sub_qubos':
                    response = solve_eigensolver_sub_qubos(Q, xplets, **solver_config)

                else:
                    raise Exception('Invalid solver name.')

            response = {
                'response': response,
                'vars_predetermined': vars_predetermined,
                'solver_config': solver_config
            }

            if solver == 'vqe_slices' or solver == 'vqe_sub_qubos':
                final_doublets, final_tracks, energy_total = process_response_vqe(response)
            elif solver == 'eigensolver_slices' or solver == 'eigensolver_sub_qubos':
                final_doublets, final_tracks, energy_total = process_response_eigensolver(response)
            elif solver == 'neal_slices':
                final_doublets, final_tracks, energy_total = process_response_neal_slices(response)
            else:
                final_doublets, final_tracks, energy = process_response(response)

        # compute scores
        p, r, ms = dw.compute_score(final_doublets)
        trackml = dw.compute_trackml_score(final_tracks)
        if solver == 'vqe_slices' or solver =='vqe_sub_qubos' or solver =='eigensolver_slices' or solver =='eigensolver_sub_qubos' or solver=='neal_slices':
            en = energy_total
        else:
            en = response['response'].record.energy[0]

        # output composition
        _, _, d_real = diff_rows(dw.get_real_doublets(), final_doublets)
        _, d_fakes, d_real_all = diff_rows(dw.get_real_doublets(with_unfocused=True), final_doublets)

        # save response
        output_filename = op.join(
            output_path.format(evt=event, ds=ds),
            unique_prefix + f'-{i}-response.pickle')

        with open(output_filename, 'wb') as f:
            pickle.dump(response, f)

        # gather stats
        mat.append(
            [
                event, ds, seed, i,
                len(final_tracks),
                len(dw.get_real_doublets()), len(final_doublets),
                len(d_real), len(d_real_all), len(d_fakes),
                p, r, trackml, len(ms),
                en, en0, en - en0,
            ] + qtime_info + time_info)


if __name__ == '__main__':
    mat = []

    for size in [8, 16, 32, 64, 128, 256, 512, 1024]:
        solver_config.update({'qubo_slice_size': size})
        for event in events:
            for ds in dss:
                print(f'\n==========>>> processing event {event} ds{ds}\n', flush=True)
                run_one(event, ds)

                headers = 'event,percent,seed,repeat,tracks_gen,' \
                          'n_true,n_gen,n_real,n_real_all,n_fakes,' \
                          'precision,recall,trackml,missings,' \
                          'en,en0,endiff,' \
                          'qtime_cpu,qtime_wall,cpu_time,wall_time'

                stats = pd.DataFrame(mat, columns=headers.split(','))
                stats_prefix = str(event) + '-' + str(ds)
                for value in solver_config.values():
                    stats_prefix += '-' + str(value)
                stats_filepath = op.join(
                    stats_path,
                    stats_prefix + '-stats.csv')
                stats.to_csv(stats_filepath, mode='a+', index=False)
