import numpy as np


class AscendingSizeCaseGenerator:

    def __init__(self, df, class_label, repetitions=1, lower=0.1, upper=0.7, step=0.1, test_split=0.3):
        self.df = df
        self.class_label = class_label
        self.lower = lower
        self.limit = upper + step
        self.step = step
        self.test_split = test_split
        self.repetitions = repetitions

        if upper + test_split > 1:
            raise ValueError("test_split + upper must be less than 1")

    def sample_data(self, sample_percent):

        grouped = self.df.groupby(self.class_label).groups

        samples_train = []
        samples_test = []

        for group, ixs in grouped.items():
            group_sample_size = int(sample_percent * len(ixs.values))
            group_train_samples = np.random.choice(ixs.values, group_sample_size, replace=False)
            samples_train.extend(group_train_samples)

            test_sample_percent = self.test_split
            ixs_test = list(set(ixs.values) - set(group_train_samples))
            group_sample_size_test = int(test_sample_percent * len(ixs.values))
            group_test_samples = np.random.choice(ixs_test, group_sample_size_test, replace=False)
            samples_test.extend(group_test_samples)

        return samples_train, samples_test

    def get_iterator(self):

        cs = np.arange(self.lower, self.limit, self.step)

        for c in cs:
            for i in range(self.repetitions):
                samples_train, samples_test = self.sample_data(c)
                yield samples_train, samples_test
