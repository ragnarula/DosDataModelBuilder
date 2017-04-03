import argparse
import errno
import importlib
import os
import tempfile
import pandas as pd
from iterative_pipeline_grid_search.core import IterativeGridSearch
import logging


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
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="The name of the python module with a pipelines object")
    parser.add_argument("data", help="The path to the input data")
    parser.add_argument("results_dir", help="The path to output the results in")
    args = parser.parse_args()

    data_path = os.path.join(os.getcwd(), args.data)
    results_path = os.path.join(os.getcwd(), args.results_dir)

    if not is_writable(results_path):
        print("Results directory {} is not writeable".format(results_path))
        return

    data = pd.read_csv(data_path)

    if args.config:
        config = importlib.import_module(args.config)
        pipelines = config.pipelines
        run_params = config.run_params
    else:
        import config
        from config import pipelines
        from config import run_params

    runner = IterativeGridSearch(run_params, pipelines, data, results_path)
    runner.run()


if __name__ == "__main__":
    main()
