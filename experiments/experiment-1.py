import argparse
import pandas as pd
import multiprocessing as mp
from kaizen import RandomShuffleCaseGenerator, Experiment1ResultGenerator
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="The name of the python module with a pipelines object")
    parser.add_argument("data", help="The path to the input data")
    args = parser.parse_args()
    data = pd.read_csv(args.data)

    data = data[data.Class != 'slowloris']

    case_generator = RandomShuffleCaseGenerator(data, 'Class', repetitions=1)
    c_values = [x ** y for x, y in zip([2] * 31, range(-15, -1, 1))]

    result_generator = Experiment1ResultGenerator(data, 'Class', 'flooding', 'linear', c_values)


    pool = mp.Pool(processes=4)
    for res in result_generator.get_iterator(pool, case_generator.get_iterator()):
        print(res)

    # results = pd.DataFrame(list(result_generator.get_iterator()))
    #
    # results.boxplot(by='c', column=['val_recall', 'train_recall', 'test_recall'])
    plt.show()

if __name__ == "__main__":
    main()
