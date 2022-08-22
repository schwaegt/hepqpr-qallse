"""
This script recreates the datasets used in the benchmarks.

To run this script:

    - download the train_100_events.zip dataset from the TrackML challenge
    - update the BUILD CONFIG options below (input paths and output paths)

"""
import os.path as op

from hepqpr.qallse.cli.func import *
from hepqpr.qallse.dsmaker import create_dataset

# ==== BUILD CONFIG

loglevel = logging.DEBUG
trackml_train_path = 'C:/Users/timsc/hepqpr-qallse/src/hepqpr/qallse/dsmaker/data/'

output_path = 'C:/Users/timsc/hepqpr-qallse-data/evt{evt}/'  # f'~/current/hpt-collapse

# ==== seeds used

ds_info = """
1000,0.05,1543636853
1001,0.05,1543636858
1002,0.05,1543636871
1003,0.05,1543636897
1004,0.05,1543636938
1005,0.05,1543637005
1006,0.05,1543637104
1007,0.05,1544857310
1008,0.05,1544857317
1009,0.05,1544857331
"""

# ==== configure logging

init_logging(logging.DEBUG, sys.stdout)

# ==== generation

headers = 'event,percent,num_hits,num_noise,num_tracks,num_important_tracks,random_seed,cpu_time,wall_time'.split(',')

if __name__ == '__main__':
    mat = []
    for row in ds_info.strip().split('\n'):
        e, d, s = row.split(',')
        event, ds, seed = int(e), float(d), int(s)
        prefix = f'ds{ds*100:.0f}'

        print(f'\n>>>> {prefix} <<<<\n')
        with time_this() as time_info:
            metas, path = create_dataset(
                density=ds,
                input_path=op.join(trackml_train_path, f'event00000{event}-hits.csv'),
                output_path=output_path.format(evt=event),
                prefix=prefix,
                min_hits_per_track=5,
                high_pt_cut=1.0,
                random_seed=int(seed),
                double_hits_ok=False,
                gen_doublets=True
            )

        mat.append([
            event,
            int(ds * 100),
            metas['num_hits'],
            metas['num_noise'],
            metas['num_tracks'],
            metas['num_important_tracks'],
            seed,
            time_info[0],
            time_info[1],
        ])

    stats = pd.DataFrame(mat, columns=headers)
    stats.to_csv('recreate_datasets.csv', index=False)
