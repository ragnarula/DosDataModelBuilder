import argparse
import errno
import itertools
import multiprocessing as mp
import tempfile

import pandas as pd

from src.result_generator import ResultGenerator
from src.case_generator import RandomShuffleCaseGenerator
from src.result_writer import CSVResultWriter

pool = mp.Pool()

params_linear = {
    'svc__C': [x ** y for x, y in zip([2] * 31, range(-15, 0, 1))],
}

params_linear_norm = {
    'svc__C': [x ** y for x, y in zip([2] * 31, range(-15, 8, 1))],
}
params_rbf = {
    'svc__C': [x ** y for x, y in zip([2] * 31, range(-15, 15, 1))],
    'svc__gamma': [x ** y for x, y in zip([2] * 31, range(-15, 16, 1))]
}

prefix_linear = 'objective-2-linear'
prefix_scaled = 'objective-2-linear-scaled'
prefix_rbf = 'objective-2-rbf'

default_results_dir = 'data/intermediate/objective-2'
default_data = 'data/external/expanded_dataset_v1.csv'


# From http://stackoverflow.com/questions/2113427/determining-whether-a-directory-is-writeable
def is_writable(path):
    try:
        testfile = tempfile.TemporaryFile(dir=path)
        testfile.close()
    except OSError as e:
        if e.errno == errno.EACCES:  # 13
            return False
        e.filename = path
        raise
    return True


def get_result_iterator(group, data, pos, reps, params, kernel, scaler=False):
    normal_flood_case_generator = RandomShuffleCaseGenerator(data, 'Class', repetitions=reps)
    normal_flood_result_generator = ResultGenerator(data, 'Class', pos, kernel, params, with_scaler=scaler)
    results = normal_flood_result_generator.get_iterator(pool, normal_flood_case_generator.get_iterator())
    results = map(lambda x: dict({'group': group}, **x), results)
    return results


def run_experiment(data, reps, kernel, params, results_dir, prefix, scaler):
    all_data = pd.read_csv(data)
    normal_flood_data = all_data[all_data.Class != 'slowloris']
    normal_slowloris_data = all_data[all_data.Class != 'flooding']
    flooding_slowloris_data = all_data[all_data.Class != 'normal']

    all_data_results = get_result_iterator('all', all_data, 'normal', reps, params, kernel, scaler=scaler)
    normal_flood_results = get_result_iterator('normal_flood', normal_flood_data, 'flooding', reps, params,
                                               kernel, scaler=scaler)
    normal_slowloris_results = get_result_iterator('normal_slowloris', normal_slowloris_data, 'slowloris',
                                                   reps, params, kernel, scaler=scaler)
    flooding_slowloris_results = get_result_iterator('flooding_slowloris', flooding_slowloris_data, 'flooding',
                                                     reps, params, kernel, scaler=scaler)

    chain = itertools.chain.from_iterable([
        all_data_results,
        normal_flood_results,
        normal_slowloris_results,
        flooding_slowloris_results
    ])

    chain = map(log, chain)

    result_writer = CSVResultWriter(chain, results_dir, prefix)
    result_writer.write()


def log(item):
    print(item)
    return item


def main():
    # logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", help="The number of times to repeat each shuffle", default=1, type=int)
    parser.add_argument("--data", help="The path to the input data", default=default_data)
    parser.add_argument("--results_dir", help="The path to output the results in", default=default_results_dir)

    args = parser.parse_args()

    if not is_writable(args.results_dir):
        print("Results directory {} is not writeable".format(args.results_dir))
        return

    run_experiment(args.data, args.reps, 'linear', params_linear, args.results_dir, prefix_linear, scaler=False)
    run_experiment(args.data, args.reps, 'linear', params_linear_norm, args.results_dir, prefix_scaled, scaler=True)
    run_experiment(args.data, args.reps, 'rbf', params_rbf, args.results_dir, prefix_rbf, scaler=True)


if __name__ == "__main__":
    main()
