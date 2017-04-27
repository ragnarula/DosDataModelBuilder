import logging
import itertools
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# class MultiPipelineResultGenerator:
#
#     def __init__(self, df, class_label, case_iterator, pipelines, n_folds=10, n_processors=-1):
#         self.df = df
#         self.case_iterator = case_iterator
#         self.pipelines = pipelines
#         self.folds = n_folds
#         self.class_label = class_label
#         self.n_processors = n_processors
#         self.logger = logging.getLogger(__name__ + ":MultiPipelineResultGenerator")
#
#     def pipeline_results(self, case):
#
#         ixs_train, ixs_test = case
#         num_training_samples = len(ixs_train)
#         self.logger.info('Starting {} sample run'.format(num_training_samples))
#
#         train = self.df.ix[ixs_train]
#         test = self.df.ix[ixs_test]
#
#         x_train = train.drop([self.class_label], axis=1)
#         y_train = train[self.class_label]
#
#         x_test = test.drop([self.class_label], axis=1)
#         y_test = test[self.class_label]
#
#         for pipe in self.pipelines:
#             self.logger.info('Running pipeline {}'.format(pipe['name']))
#             search = GridSearchCV(pipe['pipeline'],
#                                   pipe['params'],
#                                   n_jobs=self.n_processors,
#                                   scoring='f1_macro',
#                                   cv=self.folds)
#
#             classifier = search.fit(x_train, y_train)
#
#             y_pred = classifier.predict(x_test)
#
#             res = {
#                 'pipeline': pipe['name'],
#                 'classifier': classifier,
#                 'y': y_test,
#                 'y_pred': y_pred,
#                 'num_training_samples': num_training_samples
#             }
#
#             yield res
#
#     def result_metrics(self, pipeline_result):
#         cm = self.metric_from_result(pipeline_result, confusion_matrix)
#
#         r_dict = {
#             "pipeline": pipeline_result['pipeline'],
#             "accuracy": self.metric_from_result(pipeline_result, accuracy_score),
#             "f1": self.metric_from_result(pipeline_result, f1_score, average='macro'),
#             "precision": self.metric_from_result(pipeline_result, precision_score, average='macro'),
#             "recall": self.metric_from_result(pipeline_result, recall_score, average='macro'),
#             "num_training_samples": pipeline_result['num_training_samples'],
#             "best_params": str(pipeline_result['classifier'].best_params_)
#         }
#
#         for i in range(len(cm)):
#             for j in range(len(cm[0])):
#                 key = "cm_{}{}".format(i, j)
#                 value = cm[i][j]
#                 r_dict.update({key: value})
#
#         return r_dict
#
#     @staticmethod
#     def metric_from_result(res, metric_func, **kwargs):
#         return metric_func(res['y'], res['y_pred'], **kwargs)
#
#     @staticmethod
#     def flat_map(f, items):
#         return itertools.chain.from_iterable(map(f, items))
#
#     def get_iterator(self):
#         results = self.flat_map(self.pipeline_results, self.case_iterator)
#         results = map(self.result_metrics, results)
#         return results
#
#
# class Experiment1ResultGenerator:
#
#     def __init__(self, df, class_label, pos_label, kernel, c_values, with_scaler=True):
#         self.df = df
#         self.class_label = class_label
#         self.logger = logging.getLogger(__name__ + ":Experiment1ResultGenerator")
#         self.c_values = c_values
#         self.kernel = kernel
#         self.pos_label = pos_label
#         svc = SVC(kernel=kernel)
#         scaler = StandardScaler()
#
#         if with_scaler:
#             self.pipeline = Pipeline(steps=[('scaler', scaler), ('svc', svc)])
#         else:
#             self.pipeline = Pipeline(steps=[('svc', svc)])
#
#     def get_cases(self, case):
#         for c in self.c_values:
#             yield (c, case)
#
#     def fit_predict(self, c_case):
#         c, case = c_case
#         ixs_train, ixs_test = case
#         num_training_samples = len(ixs_train)
#
#         self.logger.info('Starting {} sample run'.format(num_training_samples))
#
#         train = self.df.ix[ixs_train]
#         test = self.df.ix[ixs_test]
#
#         train = train.reset_index()
#         test = test.reset_index()
#
#         x_train = train.drop([self.class_label], axis=1)
#         y_train = train[self.class_label]
#
#         x_test = test.drop([self.class_label], axis=1)
#         y_test = test[self.class_label]
#
#         # svc = SVC(C=c, kernel=self.kernel)
#         dummy = DummyClassifier()
#         dummy.fit(x_train, y_train)
#         p_dummy = dummy.predict(x_test)
#
#         kf = KFold(n_splits=10)
#
#         y_val = []
#         p_val = []
#         cv_y_train = []
#         cv_p_train = []
#
#         for fold_idx_train, fold_idx_val in kf.split(train):
#             fold_x_train = x_train.ix[fold_idx_train]
#             fold_y_train = y_train.ix[fold_idx_train]
#
#             fold_x_val = x_train.ix[fold_idx_val]
#             y_val.extend(y_train.ix[fold_idx_val])
#
#             self.pipeline.set_params(svc__C=c).fit(fold_x_train, fold_y_train)
#
#             fold_p_train = self.pipeline.predict(fold_x_train)
#
#             cv_p_train.extend(fold_p_train)
#             cv_y_train.extend(fold_y_train)
#
#             p_val.extend(self.pipeline.predict(fold_x_val))
#
#         self.pipeline.set_params(svc__C=c).fit(x_train, y_train)
#         p_test = pd.Series(self.pipeline.predict(x_test))
#
#         return self.extract_cv_metrics(cv_y_train, cv_p_train, y_val, p_val, y_test, p_test, p_dummy, c)
#
#     def extract_cv_metrics(self, y_train, p_train, y_val, p_val, y_test, p_test, p_dummy, c):
#
#         res = {
#             "train_accuracy": accuracy_score(y_train, p_train),
#             "train_f1": f1_score(y_train, p_train, average='macro', pos_label=self.pos_label),
#             "train_precision": precision_score(y_train, p_train, average='macro', pos_label=self.pos_label),
#             "train_recall": recall_score(y_train, p_train, average='macro', pos_label=self.pos_label),
#             "val_accuracy": accuracy_score(y_val, p_val),
#             "val_f1": f1_score(y_val, p_val, average='macro', pos_label=self.pos_label),
#             "val_precision": precision_score(y_val, p_val, average='macro', pos_label=self.pos_label),
#             "val_recall": recall_score(y_val, p_val, average='macro', pos_label=self.pos_label),
#             "test_accuracy": accuracy_score(y_test, p_test),
#             "test_f1": f1_score(y_test, p_test, average='macro', pos_label=self.pos_label),
#             "test_precision": precision_score(y_test, p_test, average='macro', pos_label=self.pos_label),
#             "test_recall": recall_score(y_test, p_test, average='macro', pos_label=self.pos_label),
#             "dummy_accuracy": accuracy_score(y_test, p_dummy),
#             "dummy_f1": f1_score(y_test, p_dummy, average='macro', pos_label=self.pos_label),
#             "dummy_precision": precision_score(y_test, p_dummy, average='macro', pos_label=self.pos_label),
#             "dummy_recall": recall_score(y_test, p_dummy, average='macro', pos_label=self.pos_label),
#             "c": c
#         }
#         # print(res)
#         return res
#
#     @classmethod
#     def extract_metrics(cls, result):
#
#         y_train = result['y_train']
#         p_train = result['p_train']
#         y_test = result['y_test']
#         p_test = result['p_test']
#         c = result['c']
#
#         r_dict = {
#             "train_accuracy": accuracy_score(y_train, p_train),
#             "train_f1": f1_score(y_train, p_train, average='macro'),
#             "train_precision": precision_score(y_train, p_train, average='macro'),
#             "train_recall": recall_score(y_train, p_train, average='macro'),
#             "test_accuracy": accuracy_score(y_test, p_test),
#             "test_f1": f1_score(y_test, p_test, average='macro'),
#             "test_precision": precision_score(y_test, p_test, average='macro'),
#             "test_recall": recall_score(y_test, p_test, average='macro'),
#             "c": c
#         }
#
#         cm = confusion_matrix(y_test, p_test)
#
#         for i in range(len(cm)):
#             for j in range(len(cm[0])):
#                 key = "cm_{}{}".format(i, j)
#                 value = cm[i][j]
#                 r_dict.update({key: value})
#
#         return r_dict
#
#     @staticmethod
#     def flat_map(f, items):
#         return itertools.chain.from_iterable(map(f, items))
#
#     @staticmethod
#     def pool_flat_map(pool, f, items):
#         return itertools.chain.from_iterable(pool.imap_unordered(f, items))
#
#     def get_iterator(self, pool, case_iterator):
#         results = self.flat_map(self.get_cases, case_iterator)
#         results = pool.imap_unordered(self.fit_predict, results)
#         return results
#
#
# class Experiment2ResultGenerator:
#
#     def __init__(self, df, class_label, pos_label, kernel, params, with_scaler=True):
#         self.df = df
#         self.class_label = class_label
#         self.logger = logging.getLogger(__name__ + ":Experiment2ResultGenerator")
#         self.params = params
#         self.kernel = kernel
#         self.pos_label = pos_label
#
#         svc = SVC(kernel=kernel)
#         scaler = StandardScaler()
#
#         if with_scaler:
#             self.pipeline = Pipeline(steps=[('scaler', scaler), ('svc', svc)])
#         else:
#             self.pipeline = Pipeline(steps=[('svc', svc)])
#
#     def get_cases(self, case):
#         params = (dict(zip(self.params, x)) for x in itertools.product(*self.params.values()))
#         for p in params:
#             yield (p, case)
#
#     def fit_predict(self, c_case):
#         params, case = c_case
#         ixs_train, ixs_test = case
#         num_training_samples = len(ixs_train)
#
#         self.logger.info('Starting {} sample run'.format(num_training_samples))
#
#         train = self.df.ix[ixs_train]
#         test = self.df.ix[ixs_test]
#
#         train = train.reset_index()
#         test = test.reset_index()
#
#         x_train = train.drop([self.class_label], axis=1)
#         y_train = train[self.class_label]
#
#         x_test = test.drop([self.class_label], axis=1)
#         y_test = test[self.class_label]
#
#         # svc = SVC(**params, kernel=self.kernel)
#         dummy = DummyClassifier()
#         dummy.fit(x_train, y_train)
#         p_dummy = dummy.predict(x_test)
#
#         kf = KFold(n_splits=10)
#
#         y_val = []
#         p_val = []
#         cv_y_train = []
#         cv_p_train = []
#
#         for fold_idx_train, fold_idx_val in kf.split(train):
#             fold_x_train = x_train.ix[fold_idx_train]
#             fold_y_train = y_train.ix[fold_idx_train]
#
#             fold_x_val = x_train.ix[fold_idx_val]
#             y_val.extend(y_train.ix[fold_idx_val])
#
#             self.pipeline.set_params(**params).fit(fold_x_train, fold_y_train)
#
#             fold_p_train = self.pipeline.predict(fold_x_train)
#
#             cv_p_train.extend(fold_p_train)
#             cv_y_train.extend(fold_y_train)
#
#             p_val.extend(self.pipeline.predict(fold_x_val))
#
#         self.pipeline.set_params(**params).fit(x_train, y_train)
#         p_test = pd.Series(self.pipeline.predict(x_test))
#
#         return self.extract_cv_metrics(cv_y_train, cv_p_train, y_val, p_val, y_test, p_test, p_dummy, params)
#
#     def extract_cv_metrics(self, y_train, p_train, y_val, p_val, y_test, p_test, p_dummy, params):
#
#         res = {
#             "train_accuracy": accuracy_score(y_train, p_train),
#             "train_f1": f1_score(y_train, p_train, average='macro', pos_label=self.pos_label),
#             "train_precision": precision_score(y_train, p_train, average='macro', pos_label=self.pos_label),
#             "train_recall": recall_score(y_train, p_train, average='macro', pos_label=self.pos_label),
#             "val_accuracy": accuracy_score(y_val, p_val),
#             "val_f1": f1_score(y_val, p_val, average='macro', pos_label=self.pos_label),
#             "val_precision": precision_score(y_val, p_val, average='macro', pos_label=self.pos_label),
#             "val_recall": recall_score(y_val, p_val, average='macro', pos_label=self.pos_label),
#             "test_accuracy": accuracy_score(y_test, p_test),
#             "test_f1": f1_score(y_test, p_test, average='macro', pos_label=self.pos_label),
#             "test_precision": precision_score(y_test, p_test, average='macro', pos_label=self.pos_label),
#             "test_recall": recall_score(y_test, p_test, average='macro', pos_label=self.pos_label),
#             "dummy_accuracy": accuracy_score(y_test, p_dummy),
#             "dummy_f1": f1_score(y_test, p_dummy, average='macro', pos_label=self.pos_label),
#             "dummy_precision": precision_score(y_test, p_dummy, average='macro', pos_label=self.pos_label),
#             "dummy_recall": recall_score(y_test, p_dummy, average='macro', pos_label=self.pos_label)
#         }
#         res.update(params)
#         return res
#
#     @staticmethod
#     def flat_map(f, items):
#         return itertools.chain.from_iterable(map(f, items))
#
#     @staticmethod
#     def pool_flat_map(pool, f, items):
#         return itertools.chain.from_iterable(pool.imap_unordered(f, items))
#
#     def get_iterator(self, pool, case_iterator):
#         results = self.flat_map(self.get_cases, case_iterator)
#         results = pool.imap_unordered(self.fit_predict, results)
#         return results
#
#
# class Experiment3ResultGenerator:
#
#     def __init__(self, df, class_label, pos_label, kernel, params, with_scaler=True):
#         self.df = df
#         self.class_label = class_label
#         self.logger = logging.getLogger(__name__ + ":Experiment3ResultGenerator")
#         self.params = params
#         self.kernel = kernel
#         self.pos_label = pos_label
#
#         svc = SVC(kernel=kernel)
#         scaler = StandardScaler()
#
#         if with_scaler:
#             self.pipeline = Pipeline(steps=[('scaler', scaler), ('svc', svc)])
#         else:
#             self.pipeline = Pipeline(steps=[('svc', svc)])
#
#     def get_cases(self, case):
#         params = (dict(zip(self.params, x)) for x in itertools.product(*self.params.values()))
#         for p in params:
#             yield (p, case)
#
#     def fit_predict(self, c_case):
#         params, case = c_case
#         ixs_train, ixs_test = case
#         num_training_samples = len(ixs_train)
#
#         self.logger.info('Starting {} sample run'.format(num_training_samples))
#
#         train = self.df.ix[ixs_train]
#         test = self.df.ix[ixs_test]
#
#         train = train.reset_index()
#         test = test.reset_index()
#
#         x_train = train.drop([self.class_label], axis=1)
#         y_train = train[self.class_label]
#
#         x_test = test.drop([self.class_label], axis=1)
#         y_test = test[self.class_label]
#
#         # svc = SVC(**params, kernel=self.kernel)
#         dummy = DummyClassifier()
#         dummy.fit(x_train, y_train)
#         p_dummy = dummy.predict(x_test)
#
#         kf = KFold(n_splits=10)
#
#         y_val = []
#         p_val = []
#         cv_y_train = []
#         cv_p_train = []
#
#         for fold_idx_train, fold_idx_val in kf.split(train):
#             fold_x_train = x_train.ix[fold_idx_train]
#             fold_y_train = y_train.ix[fold_idx_train]
#
#             fold_x_val = x_train.ix[fold_idx_val]
#             y_val.extend(y_train.ix[fold_idx_val])
#
#             self.pipeline.set_params(**params).fit(fold_x_train, fold_y_train)
#
#             fold_p_train = self.pipeline.predict(fold_x_train)
#
#             cv_p_train.extend(fold_p_train)
#             cv_y_train.extend(fold_y_train)
#
#             p_val.extend(self.pipeline.predict(fold_x_val))
#
#         self.pipeline.set_params(**params).fit(x_train, y_train)
#         p_test = pd.Series(self.pipeline.predict(x_test))
#
#         metrics = self.extract_cv_metrics(cv_y_train, cv_p_train, y_val, p_val, y_test, p_test, p_dummy, params)
#
#         metrics.update({'train_size': len(train.index), 'test_size': len(test.index)})
#
#         return metrics
#
#     def extract_cv_metrics(self, y_train, p_train, y_val, p_val, y_test, p_test, p_dummy, params):
#
#         res = {
#             "train_accuracy": accuracy_score(y_train, p_train),
#             "train_f1": f1_score(y_train, p_train, average='macro', pos_label=self.pos_label),
#             "train_precision": precision_score(y_train, p_train, average='macro', pos_label=self.pos_label),
#             "train_recall": recall_score(y_train, p_train, average='macro', pos_label=self.pos_label),
#             "val_accuracy": accuracy_score(y_val, p_val),
#             "val_f1": f1_score(y_val, p_val, average='macro', pos_label=self.pos_label),
#             "val_precision": precision_score(y_val, p_val, average='macro', pos_label=self.pos_label),
#             "val_recall": recall_score(y_val, p_val, average='macro', pos_label=self.pos_label),
#             "test_accuracy": accuracy_score(y_test, p_test),
#             "test_f1": f1_score(y_test, p_test, average='macro', pos_label=self.pos_label),
#             "test_precision": precision_score(y_test, p_test, average='macro', pos_label=self.pos_label),
#             "test_recall": recall_score(y_test, p_test, average='macro', pos_label=self.pos_label),
#             "dummy_accuracy": accuracy_score(y_test, p_dummy),
#             "dummy_f1": f1_score(y_test, p_dummy, average='macro', pos_label=self.pos_label),
#             "dummy_precision": precision_score(y_test, p_dummy, average='macro', pos_label=self.pos_label),
#             "dummy_recall": recall_score(y_test, p_dummy, average='macro', pos_label=self.pos_label)
#         }
#         res.update(params)
#         return res
#
#     @staticmethod
#     def flat_map(f, items):
#         return itertools.chain.from_iterable(map(f, items))
#
#     @staticmethod
#     def pool_flat_map(pool, f, items):
#         return itertools.chain.from_iterable(pool.imap_unordered(f, items))
#
#     def get_iterator(self, pool, case_iterator):
#         results = self.flat_map(self.get_cases, case_iterator)
#         results = pool.imap_unordered(self.fit_predict, results)
#         return results


class ResultGenerator:

    def __init__(self, df, class_label, pos_label, kernel, params, with_scaler=True):
        self.df = df
        self.class_label = class_label
        self.logger = logging.getLogger(__name__ + ":ResultGenerator")
        self.params = params
        self.kernel = kernel
        self.pos_label = pos_label

        svc = SVC(kernel=kernel)
        scaler = StandardScaler()

        if with_scaler:
            self.pipeline = Pipeline(steps=[('scaler', scaler), ('svc', svc)])
        else:
            self.pipeline = Pipeline(steps=[('svc', svc)])

    def get_cases(self, case):
        params = (dict(zip(self.params, x)) for x in itertools.product(*self.params.values()))
        for p in params:
            yield (p, case)

    def fit_predict(self, c_case):
        params, case = c_case
        ixs_train, ixs_test = case
        num_training_samples = len(ixs_train)

        self.logger.info('Starting {} sample run'.format(num_training_samples))

        train = self.df.ix[ixs_train]
        test = self.df.ix[ixs_test]

        train = train.reset_index()
        test = test.reset_index()

        x_train = train.drop([self.class_label], axis=1)
        y_train = train[self.class_label]

        x_test = test.drop([self.class_label], axis=1)
        y_test = test[self.class_label]

        # svc = SVC(**params, kernel=self.kernel)
        dummy = DummyClassifier()
        dummy.fit(x_train, y_train)
        p_dummy = dummy.predict(x_test)

        kf = KFold(n_splits=10)

        y_val = []
        p_val = []
        cv_y_train = []
        cv_p_train = []

        for fold_idx_train, fold_idx_val in kf.split(train):
            fold_x_train = x_train.ix[fold_idx_train]
            fold_y_train = y_train.ix[fold_idx_train]

            fold_x_val = x_train.ix[fold_idx_val]
            y_val.extend(y_train.ix[fold_idx_val])

            self.pipeline.set_params(**params).fit(fold_x_train, fold_y_train)

            fold_p_train = self.pipeline.predict(fold_x_train)

            cv_p_train.extend(fold_p_train)
            cv_y_train.extend(fold_y_train)

            p_val.extend(self.pipeline.predict(fold_x_val))

        self.pipeline.set_params(**params).fit(x_train, y_train)
        p_test = pd.Series(self.pipeline.predict(x_test))

        metrics = self.extract_cv_metrics(cv_y_train, cv_p_train, y_val, p_val, y_test, p_test, p_dummy, params)

        metrics.update({'train_size': len(train.index), 'test_size': len(test.index)})

        return metrics

    def extract_cv_metrics(self, y_train, p_train, y_val, p_val, y_test, p_test, p_dummy, params):

        res = {
            "train_accuracy": accuracy_score(y_train, p_train),
            "train_f1": f1_score(y_train, p_train, average='macro', pos_label=self.pos_label),
            "train_precision": precision_score(y_train, p_train, average='macro', pos_label=self.pos_label),
            "train_recall": recall_score(y_train, p_train, average='macro', pos_label=self.pos_label),
            "val_accuracy": accuracy_score(y_val, p_val),
            "val_f1": f1_score(y_val, p_val, average='macro', pos_label=self.pos_label),
            "val_precision": precision_score(y_val, p_val, average='macro', pos_label=self.pos_label),
            "val_recall": recall_score(y_val, p_val, average='macro', pos_label=self.pos_label),
            "test_accuracy": accuracy_score(y_test, p_test),
            "test_f1": f1_score(y_test, p_test, average='macro', pos_label=self.pos_label),
            "test_precision": precision_score(y_test, p_test, average='macro', pos_label=self.pos_label),
            "test_recall": recall_score(y_test, p_test, average='macro', pos_label=self.pos_label),
            "dummy_accuracy": accuracy_score(y_test, p_dummy),
            "dummy_f1": f1_score(y_test, p_dummy, average='macro', pos_label=self.pos_label),
            "dummy_precision": precision_score(y_test, p_dummy, average='macro', pos_label=self.pos_label),
            "dummy_recall": recall_score(y_test, p_dummy, average='macro', pos_label=self.pos_label)
        }
        res.update(params)
        return res

    @staticmethod
    def flat_map(f, items):
        return itertools.chain.from_iterable(map(f, items))

    @staticmethod
    def pool_flat_map(pool, f, items):
        return itertools.chain.from_iterable(pool.imap_unordered(f, items))

    def get_iterator(self, pool, case_iterator):
        results = self.flat_map(self.get_cases, case_iterator)
        results = pool.imap_unordered(self.fit_predict, results)
        return results
