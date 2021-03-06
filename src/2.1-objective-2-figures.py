import argparse
import errno
import tempfile
import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



default_data_dir = 'data/intermediate/objective-2'
default_out_dir = 'figures/objective-2'


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


def main():
    # logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="The data dir containing the csv of intermediate data", default=default_data_dir)
    parser.add_argument("--out", help="The results dir where the figs will be places", default=default_out_dir)

    args = parser.parse_args()

    if not is_writable(args.out):
        print("Results directory {} is not writeable".format(args.results_dir))
        return

    # results_dir = args.results_dir

    # results_dir = 'experiment-1_2017-04-19_14:37:49.692977'
    all_csv_files = glob.glob(args.data + "/*linear*.csv")

    train_val_metrics = [
        "train_accuracy",
        "train_f1",
        "train_precision",
        "train_recall",
        "val_accuracy",
        "val_f1",
        "val_precision",
        "val_recall",
    ]

    test_dummy_metrics = [
        "test_accuracy",
        "test_f1",
        "test_precision",
        "test_recall",
        "dummy_accuracy",
        "dummy_f1",
        "dummy_precision",
        "dummy_recall"
    ]

    for file in all_csv_files:

        file_name = os.path.basename(file)
        file_name = os.path.splitext(file_name)[0]
        title, data_set_name = file_name.split('_', 1)
        prefix = title
        title = title.replace('-', ' ').title()
        data_set_tile = data_set_name.replace('_', ' ').title()
        data_set_tile = data_set_tile.replace(' ', ' vs ')
        os.makedirs(os.path.join(args.out, data_set_name), exist_ok=True)

        for metric in train_val_metrics:

            os.makedirs(os.path.join(args.out, data_set_name), exist_ok=True)
            fig_file_name = '{}_{}_{}.png'.format(prefix, data_set_name, metric)

            print(fig_file_name)
            df = pd.read_csv(file, index_col=None, header=0)
            both = df.boxplot(by='svc__C', column=[metric], return_type='both')

            ax, lines = both[0]
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

            plt.suptitle('{} - {}'.format(title, data_set_tile))
            ax.set_xlabel('C')
            fig = ax.get_figure()
            fig.tight_layout()
            plt.subplots_adjust(top=0.9)
            fig.savefig(os.path.join(args.out, data_set_name, fig_file_name))
            plt.close()

        df = pd.read_csv(file, index_col=None, header=0)
        grouped = df.groupby(df.svc__C)

        for name, group in grouped:
            fig_file_name = '{}_{}_{}_c={}.png'.format(prefix, data_set_name, 'test_metrics', name)

            print(fig_file_name)
            ax, lines = group.boxplot(column=test_dummy_metrics, return_type='both')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

            ax.set_title('C = {}'.format(str(name)))
            plt.suptitle('{} - {} Test Metrics'.format(title, data_set_tile))

            fig = ax.get_figure()
            fig.tight_layout()
            plt.subplots_adjust(top=0.9)
            fig.savefig(os.path.join(args.out, data_set_name, fig_file_name))
            plt.close()

    all_csv_files = glob.glob(args.data + "/*rbf*.csv")

    ticks = [x ** y for x, y in zip([2] * 31, range(-15, 15, 1))]
    for file in all_csv_files:

        file_name = os.path.basename(file)
        file_name = os.path.splitext(file_name)[0]
        title, data_set_name = file_name.split('_', 1)
        prefix = title
        title = title.replace('-', ' ').title()
        data_set_tile = data_set_name.replace('_', ' ').title()
        data_set_tile = data_set_tile.replace(' ', ' vs ')
        os.makedirs(os.path.join(args.out, data_set_name), exist_ok=True)

        for metric in train_val_metrics:

            os.makedirs(os.path.join(args.out, data_set_name), exist_ok=True)
            fig_file_name = '{}_{}_{}.png'.format(prefix, data_set_name, metric)
            print(fig_file_name)

            df = pd.read_csv(file, index_col=None, header=0)

            pivoted = df.pivot_table(index='svc__C', columns='svc__gamma', values=metric)
            # pivoted.columns = np.log2(pivoted.columns)
            # pivoted.index = np.log2(pivoted.index)

            X, Y = np.meshgrid(np.log2(pivoted.columns), np.log2(pivoted.index))

            fig = plt.figure()

            plt.suptitle('{} - {} data - {}'.format(title, data_set_tile, metric.replace('_', ' ').title()))
            ax = Axes3D(fig)

            ax.set_xlabel('log2 gamma')
            ax.set_ylabel('log2 C')
            ax.set_zlabel(metric.replace('_', ' ').title())
            surf = ax.plot_surface(X, Y, pivoted, rstride=1, cstride=1, cmap='rainbow',
                                   linewidth=0, antialiased=False)

            # plt.show()
            fig.savefig(os.path.join(args.out, data_set_name, fig_file_name))
            plt.close()

            max = np.argmax(pivoted.as_matrix())
            c_max, gamma_max = np.unravel_index(max, pivoted.as_matrix().shape)

            c_max = pivoted.index[c_max]
            gamma_max = pivoted.columns[gamma_max]

            filtered = df[(df.svc__C == c_max) & (df.svc__gamma == gamma_max)]

            ax, lines = filtered.boxplot(column=test_dummy_metrics, return_type='both')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.set_title('C = {}, gamma = {}'.format(str(c_max), str(gamma_max)))
            plt.suptitle('{} - {} Test Metrics'.format(title, data_set_tile))
            fig = ax.get_figure()
            fig.tight_layout()
            plt.subplots_adjust(top=0.9)

            fig_file_name = '{}_{}_{}_best={}.png'.format(prefix, data_set_name, 'test_metrics', metric)
            print(fig_file_name)
            # plt.show()
            fig.savefig(os.path.join(args.out, data_set_name, fig_file_name))
            plt.close()
if __name__ == "__main__":
    main()
