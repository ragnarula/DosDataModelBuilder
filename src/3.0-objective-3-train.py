import argparse
import datetime
import errno
import itertools
import multiprocessing as mp
import os
import tempfile

import pandas as pd

from src.result_generator import ResultGenerator
from src.case_generator import AscendingSizeCaseGenerator
from src.result_writer import CSVResultWriter
pool = mp.Pool()

params = {
    'svc__C': [256],
    'svc__gamma': [0.125]
}

prefix = 'objective-3'

default_results_dir = 'data/intermediate/objective-3'
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


def get_result_iterator(group, data, pos, reps):
    normal_flood_case_generator = AscendingSizeCaseGenerator(data, 'Class', repetitions=reps)
    normal_flood_result_generator = ResultGenerator(data, 'Class', pos, 'rbf', params)
    results = normal_flood_result_generator.get_iterator(pool, normal_flood_case_generator.get_iterator())
    results = map(lambda x: dict({'group': group}, **x), results)
    return results


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

    all_data = pd.read_csv(args.data)
    normal_flood_data = all_data[all_data.Class != 'slowloris']
    normal_slowloris_data = all_data[all_data.Class != 'flooding']
    flooding_slowloris_data = all_data[all_data.Class != 'normal']

    all_data_results = get_result_iterator('all', all_data, 'normal', args.reps)
    normal_flood_results = get_result_iterator('normal_flood', normal_flood_data, 'flooding', args.reps)
    normal_slowloris_results = get_result_iterator('normal_slowloris', normal_slowloris_data, 'slowloris', args.reps)
    flooding_slowloris_results = get_result_iterator('flooding_slowloris', flooding_slowloris_data, 'flooding', args.reps)

    chain = itertools.chain.from_iterable([
        all_data_results,
        normal_flood_results,
        normal_slowloris_results,
        flooding_slowloris_results
    ])

    chain = map(log, chain)

    result_writer = CSVResultWriter(chain, args.results_dir, prefix)
    result_writer.write()


if __name__ == "__main__":
    main()
