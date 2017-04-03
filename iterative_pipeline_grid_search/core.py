import os
import pandas as pd
import numpy as np
import datetime
import logging
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class IterativeGridSearch:

    def __init__(self, run_params, pipes, data, results):

        self.logger = logging.getLogger(__name__ + '.IterativeGridSearch')
        self.results_dir = os.path.join(results, "results_{}".format(datetime.datetime.utcnow()).replace(' ', '_'))
        self.data = data
        self.run_params = run_params
        self.sample_rows = []
        self.pipes = pipes
        self.result_rows = {}

    def sample_data(self, sample_percent):

        grouped = self.data.groupby(['Class']).groups
        samples_train = []
        samples_test = []

        for group, ixs in grouped.items():

            group_sample_size = int(sample_percent * len(ixs.values))
            group_train_samples = np.random.choice(ixs.values, group_sample_size, replace=False)
            samples_train.extend(group_train_samples)

            test_sample_percent = self.run_params.get('test', 0.3)
            ixs_test = list(set(ixs.values) - set(group_train_samples))
            group_sample_size_test = int(test_sample_percent * len(ixs_test))
            group_test_samples = np.random.choice(ixs_test, group_sample_size_test, replace=False)
            samples_test.extend(group_test_samples)

        return samples_train, samples_test

    def create_row_samples(self):

        self.logger.info('Started creating row samples at {}'.format(datetime.datetime.utcnow()))
        lower = self.run_params.get('lower', 0.1)
        step = self.run_params.get('step', 0.1)
        upper = self.run_params.get('upper', 0.7) + step

        cs = np.arange(lower, upper, step)
        for c in cs:
            for i in range(self.run_params.get('repetitions', 1)):
                samples_train, samples_test = self.sample_data(c)
                self.sample_rows.append((c, samples_train, samples_test))

        self.logger.info('Finished creating row samples at {}'.format(datetime.datetime.utcnow()))

    def create_result_dir(self):

        if not os.path.exists(self.results_dir):
            self.logger.info('Creating results dir')
            os.mkdir(self.results_dir, mode=0o755)
        else:
            self.logger.error('Results path already existed')
            raise RuntimeError('Results dir already exists')

    def run(self):

        self.create_result_dir()
        self.create_row_samples()

        for sample in self.sample_rows:
            self.run_pipelines(sample)

        self.flush(force=True)

    def run_pipelines(self, c_ixs):

        c, ixs_train, ixs_test = c_ixs
        self.logger.info('Starting {} sample run'.format(c))

        train = self.data.ix[ixs_train]
        test = self.data.ix[ixs_test]

        x_train = train.drop(['Class'], axis=1)
        y_train = train['Class']

        x_test = test.drop(['Class'], axis=1)
        y_test = test['Class']

        results = []

        for pipe in self.pipes:

            search = GridSearchCV(pipe['pipeline'],
                                  pipe['params'],
                                  n_jobs=-1,
                                  scoring='f1_macro',
                                  cv=self.run_params['folds'])

            classifier = search.fit(x_train, y_train)

            y_pred = classifier.predict(x_test)

            res = {
                'pipeline': pipe['name'],
                'classifier': classifier,
                'y': y_test,
                'y_pred': y_pred,
                'c': c
            }

            results.append(res)

        for res in results:
            self.store_result(res)

        self.logger.info('Finished {} sample run'.format(c))

    def store_result(self, r):

        cm = self.metric_from_result(r, confusion_matrix)
        r_dict = {
            "pipeline": r['pipeline'],
            "accuracy": self.metric_from_result(r, accuracy_score),
            "f1": self.metric_from_result(r, f1_score, average='macro'),
            "precision": self.metric_from_result(r, precision_score, average='macro'),
            "recall": self.metric_from_result(r, recall_score, average='macro'),
            "c": r['c'],
            "best_params": str(r['classifier'].best_params_)
        }

        for i in range(len(cm)):
            for j in range(len(cm[0])):
                key = "cm_{}{}".format(i, j)
                value = cm[i][j]
                r_dict.update({key: value})

        self.result_rows.setdefault(r['pipeline'], []).append(r_dict)
        self.flush()

    def flush(self, force=False):

        for pipe, res in self.result_rows.items():

            if len(res) > 100 or force:
                header = True
                file_path = os.path.join(self.results_dir, "results_{}.csv".format(pipe))

                self.logger.info('Flushing results to file {}'.format(file_path))

                if os.path.exists(file_path):
                    header = False

                with(open(file_path, "a+")) as f:
                    df = pd.DataFrame(res)
                    df.to_csv(f, header=header, index=False)

    @staticmethod
    def metric_from_result(res, metric_func, **kwargs):
        return metric_func(res['y'], res['y_pred'], **kwargs)