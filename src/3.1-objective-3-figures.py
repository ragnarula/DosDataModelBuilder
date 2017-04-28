import argparse
import errno
import tempfile
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt


default_data_dir = 'data/intermediate/objective-3'
default_out_dir = 'figures/objective-3'


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
        print("Results directory {} is not writeable".format(args.out))
        return

    results_dir = args.out

    # results_dir = 'experiment-1_2017-04-19_14:37:49.692977'
    all_csv_files = glob.glob(args.data + "/*.csv")

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
            both = df.boxplot(by='train_size', column=[metric], return_type='both')

            ax, lines = both[0]
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

            plt.suptitle('{} - {}'.format(title, data_set_tile))

            fig = ax.get_figure()
            fig.tight_layout()
            plt.subplots_adjust(top=0.9)
            # plt.show()
            fig.savefig(os.path.join(args.out, data_set_name, fig_file_name))
            plt.close()

        df = pd.read_csv(file, index_col=None, header=0)
        grouped = df.groupby(df.train_size)

        for name, group in grouped:
            fig_file_name = '{}_{}_{}_train_size={}.png'.format(prefix, data_set_name, 'test_metrics', name)

            print(fig_file_name)
            ax, lines = group.boxplot(column=test_dummy_metrics, return_type='both')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

            ax.set_title('Train set size = {}'.format(str(name)))
            plt.suptitle('{} - {} Test Metrics'.format(title, data_set_tile))

            fig = ax.get_figure()
            fig.tight_layout()
            plt.subplots_adjust(top=0.9)
            # plt.show()
            fig.savefig(os.path.join(args.out, data_set_name, fig_file_name))
            plt.close()


if __name__ == "__main__":
    main()
