import argparse
import errno
import importlib
import os
import tempfile
import pandas as pd
import logging
import kaizen


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
    parser.add_argument("class_label", help="The label of the row containing the class names")
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
    else:
        import config

    case_generator = kaizen.AscendingSizeCaseGenerator(df=data,
                                                       class_label=args.class_label,
                                                       repetitions=config.run_params['repetitions'],
                                                       lower=config.run_params['lower'],
                                                       upper=config.run_params['upper'],
                                                       step=config.run_params['step'],
                                                       test_split=config.run_params['test'])

    result_generator = kaizen.MultiPipelineResultGenerator(df=data,
                                                           class_label=args.class_label,
                                                           case_iterator=case_generator.get_iterator(),
                                                           pipelines=config.pipelines,
                                                           n_folds=config.run_params['folds'],
                                                           n_processors=config.run_params['threads'])

    csv_writer = kaizen.CSVPerPipelineResultWriter(result_iterator=result_generator.get_iterator(),
                                                   results_dir=args.results_dir)

    csv_writer.write()


if __name__ == "__main__":
    main()
