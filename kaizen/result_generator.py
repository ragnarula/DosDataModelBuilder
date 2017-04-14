import logging
import itertools
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class MultiPipelineResultGenerator:

    def __init__(self, df, class_label, case_iterator, pipelines, n_folds=10):
        self.df = df
        self.case_iterator = case_iterator
        self.pipelines = pipelines
        self.folds = n_folds
        self.class_label = class_label
        self.logger = logging.getLogger(__name__ + ":MultiPipelineResultGenerator")

    def pipeline_results(self, case):

        ixs_train, ixs_test = case
        num_training_samples = len(ixs_train)
        self.logger.info('Starting {} sample run'.format(num_training_samples))

        train = self.df.ix[ixs_train]
        test = self.df.ix[ixs_test]

        x_train = train.drop([self.class_label], axis=1)
        y_train = train[self.class_label]

        x_test = test.drop([self.class_label], axis=1)
        y_test = test[self.class_label]

        for pipe in self.pipelines:
            self.logger.info('Running pipeline {}'.format(pipe['name']))
            search = GridSearchCV(pipe['pipeline'],
                                  pipe['params'],
                                  n_jobs=-1,
                                  scoring='f1_macro',
                                  cv=self.folds)

            classifier = search.fit(x_train, y_train)

            y_pred = classifier.predict(x_test)

            res = {
                'pipeline': pipe['name'],
                'classifier': classifier,
                'y': y_test,
                'y_pred': y_pred,
                'num_training_samples': num_training_samples
            }

            yield res

    def result_metrics(self, pipeline_result):
        cm = self.metric_from_result(pipeline_result, confusion_matrix)

        r_dict = {
            "pipeline": pipeline_result['pipeline'],
            "accuracy": self.metric_from_result(pipeline_result, accuracy_score),
            "f1": self.metric_from_result(pipeline_result, f1_score, average='macro'),
            "precision": self.metric_from_result(pipeline_result, precision_score, average='macro'),
            "recall": self.metric_from_result(pipeline_result, recall_score, average='macro'),
            "num_training_samples": pipeline_result['num_training_samples'],
            "best_params": str(pipeline_result['classifier'].best_params_)
        }

        for i in range(len(cm)):
            for j in range(len(cm[0])):
                key = "cm_{}{}".format(i, j)
                value = cm[i][j]
                r_dict.update({key: value})

        return r_dict

    @staticmethod
    def metric_from_result(res, metric_func, **kwargs):
        return metric_func(res['y'], res['y_pred'], **kwargs)

    @staticmethod
    def flat_map(f, items):
        return itertools.chain.from_iterable(map(f, items))

    def get_iterator(self):
        results = self.flat_map(self.pipeline_results, self.case_iterator)
        results = map(self.result_metrics, results)
        return results
