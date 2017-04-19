import argparse
import errno
import tempfile
import itertools
import datetime
import pandas as pd
import glob
import os
import multiprocessing as mp
from kaizen import RandomShuffleCaseGenerator, Experiment1ResultGenerator, CSVResultWriter
import matplotlib.pyplot as plt


pool = mp.Pool()
c_values = [x ** y for x, y in zip([2] * 31, range(-15, 3, 1))]
reps = 1
prefix = 'experiment-1'

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


def get_result_iterator(group, data, pos):
    normal_flood_case_generator = RandomShuffleCaseGenerator(data, 'Class', repetitions=reps)
    normal_flood_result_generator = Experiment1ResultGenerator(data, 'Class', pos, 'linear', c_values)
    results = normal_flood_result_generator.get_iterator(pool, normal_flood_case_generator.get_iterator())
    results = map(lambda x: dict({'group': group}, **x), results)
    return results


def log(item):
    print(item)
    return item


def main():
    # logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="The name of the python module with a pipelines object")
    parser.add_argument("data", help="The path to the input data")
    parser.add_argument("results_dir", help="The path to output the results in")

    args = parser.parse_args()

    if not is_writable(args.results_dir):
        print("Results directory {} is not writeable".format(args.results_dir))
        return

    results_dir = os.path.join(args.results_dir, "{}_{}".format(prefix, datetime.datetime.utcnow()).replace(' ', '_'))

    all_data = pd.read_csv(args.data)
    normal_flood_data = all_data[all_data.Class != 'slowloris']
    normal_slowloris_data = all_data[all_data.Class != 'flooding']
    flooding_slowloris_data = all_data[all_data.Class != 'normal']

    all_data_results = get_result_iterator('all', all_data, 'normal')
    normal_flood_results = get_result_iterator('normal_flood', normal_flood_data, 'flooding')
    normal_slowloris_results = get_result_iterator('normal_slowloris', normal_slowloris_data, 'slowloris')
    flooding_slowloris_results = get_result_iterator('flooding_slowloris', flooding_slowloris_data, 'flooding')

    chain = itertools.chain.from_iterable([
        all_data_results,
        normal_flood_results,
        normal_slowloris_results,
        flooding_slowloris_results
    ])

    chain = map(log, chain)

    result_writer = CSVResultWriter(chain, results_dir, prefix)
    result_writer.write()

    # Moved to separate script

    # results_dir = 'experiment-1_2017-04-19_14:37:49.692977'
    # all_csv_files = glob.glob(results_dir + "/*.csv")
    #
    # train_val_metrics = [
    #     "train_accuracy",
    #     "train_f1",
    #     "train_precision",
    #     "train_recall",
    #     "val_accuracy",
    #     "val_f1",
    #     "val_precision",
    #     "val_recall",
    # ]
    #
    # test_dummy_metrics = [
    #     "test_accuracy",
    #     "test_f1",
    #     "test_precision",
    #     "test_recall",
    #     "dummy_accuracy",
    #     "dummy_f1",
    #     "dummy_precision",
    #     "dummy_recall"
    # ]
    #
    # for file in all_csv_files:
    #
    #     file_name = os.path.basename(file)
    #     file_name = os.path.splitext(file_name)[0]
    #     title, data_set_name = file_name.split('_', 1)
    #     title = title.replace('-', ' ').title()
    #     data_set_tile = data_set_name.replace('_', ' ').title()
    #     data_set_tile = data_set_tile.replace(' ', ' vs ')
    #     os.makedirs(os.path.join(results_dir, 'figs', data_set_name), exist_ok=True)
    #
    #     for metric in train_val_metrics:
    #
    #         os.makedirs(os.path.join(results_dir, 'figs', data_set_name), exist_ok=True)
    #         fig_file_name = '{}_{}_{}.png'.format(prefix, data_set_name, metric)
    #
    #         print(fig_file_name)
    #         df = pd.read_csv(file, index_col=None, header=0)
    #         both = df.boxplot(by='c', column=[metric], return_type='both')
    #
    #         ax, lines = both[0]
    #         ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    #
    #         plt.suptitle('{} - {}'.format(title, data_set_tile))
    #
    #         fig = ax.get_figure()
    #         fig.tight_layout()
    #         plt.subplots_adjust(top=0.9)
    #         fig.savefig(os.path.join(results_dir, 'figs', data_set_name, fig_file_name))
    #         plt.clf()
    #
    #     df = pd.read_csv(file, index_col=None, header=0)
    #     grouped = df.groupby(df.c)
    #
    #     for name, group in grouped:
    #         fig_file_name = '{}_{}_{}_c={}.png'.format(prefix, data_set_name, 'test_metrics', name)
    #
    #         print(fig_file_name)
    #         ax, lines = group.boxplot(column=test_dummy_metrics, return_type='both')
    #         ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    #
    #         ax.set_title('C = {}'.format(str(name)))
    #         plt.suptitle('{} - {} Test Metrics'.format(title, data_set_tile))
    #
    #         fig = ax.get_figure()
    #         fig.tight_layout()
    #         plt.subplots_adjust(top=0.9)
    #         fig.savefig(os.path.join(results_dir, 'figs', data_set_name, fig_file_name))
    #         plt.clf()


if __name__ == "__main__":
    main()
