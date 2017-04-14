from sklearn.pipeline import Pipeline
from sklearn import preprocessing, svm
from sklearn.decomposition import PCA

svc_linear_params = [
    {
        'svm__kernel': ['linear'],
        # 'svm__gamma': [x ** y for x, y in zip([2] * 31, range(-15, 16, 1))],
        'svm__C': [x ** y for x, y in zip([2] * 31, range(-15, 16, 1))]
    }
]

pca_svc_linear_params = [
    {
        'pca__n_components': [2, 3, 4, 5, 6],
        'pca__whiten': [True],
        'svm__kernel': ['linear'],
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

normalize_svm_linear = Pipeline(steps=[
    ('normalize', preprocessing.Normalizer()),
    ('svm', svm.SVC())
])

normalize_pca_svm_linear_pipeline = Pipeline(steps=[
    ('normalize', preprocessing.Normalizer()),
    ('pca', PCA()),
    ('svm', svm.SVC())
])

pipelines = [
    {
        "pipeline": normalize_svm_linear,
        "params": svc_linear_params,
        "name": "normalize_svm_linear"
    },
    {
        "pipeline": normalize_pca_svm_linear_pipeline,
        "params": pca_svc_linear_params,
        "name": "normalize_pca_svm_linear_pipeline"
    }
]

run_params = {
    'lower': 0.1,
    'upper': 0.7,
    'step': 0.1,
    'test': 0.3,
    'repetitions': 1,
    'folds': 10
}