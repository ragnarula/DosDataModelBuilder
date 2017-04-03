from sklearn.pipeline import Pipeline
from sklearn import preprocessing, svm
from sklearn.naive_bayes import GaussianNB

svc_params = [
    {
        'svm__kernel': ['rbf'],
        'svm__gamma': [x ** y for x, y in zip([2] * 31, range(-15, 16, 1))],
        'svm__C': [x ** y for x, y in zip([2] * 31, range(-15, 16, 1))]
    }
]

nb_params = [
    {
        'nb__priors': [None]
    }
]

params_test = [
    {
        'svm__kernel': ['rbf'],
        'svm__gamma': [1],
        'svm__C': [1]
    }
]

normalize_svm_pipeline = Pipeline(steps=[('normalize', preprocessing.Normalizer()), ('svm', svm.SVC())])
normalize_nb_pipeline = Pipeline(steps=[('normalize', preprocessing.Normalizer()), ('nb', GaussianNB())])

pipelines = [
    {
        "pipeline": normalize_svm_pipeline,
        "params": svc_params,
        "name": "normalize_svm_grid"
    },
    # {
    #     "pipeline": normalize_nb_pipeline,
    #     "params": nb_params,
    #     "name": "normalize_naive_bayes"
    # }
]

run_params = {
    'lower': 0.1,
    'upper': 0.7,
    'step': 0.1,
    'test': 0.3,
    'repetitions': 100,
    'folds': 10
}