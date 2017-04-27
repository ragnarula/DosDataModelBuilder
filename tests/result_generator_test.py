import unittest

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.kaizen import MultiPipelineResultGenerator


class MultiPipelineResultGeneratorTest(unittest.TestCase):

    cases = iter([
        ([1, 9, 2, 8], [3, 4, 7, 8]),
        ([1, 9, 2, 8], [3, 4, 7, 8])
    ])

    data = pd.DataFrame.from_dict({
        'x': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        'y': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        'Class': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    })

    test_pipe = {
        'pipeline': Pipeline(steps=[('log', LogisticRegression())]),
        'params': [
            {
                'log__C': [1.0]
            }
        ],
        'name': 'test_pipeline'
    }

    def test_should_contain_correct_columns_and_number_of_results(self):
        """
        Each item in the result iterator should contain the same columns
        There should be (num_cases * num_pipelines) results
        :return: 
        """

        fixture = MultiPipelineResultGenerator(self.data, 'Class', self.cases, [self.test_pipe, self.test_pipe], n_folds=2)
        res = list(fixture.get_iterator())
        self.assertEqual(len(res), 4)
        for i in res:
            self.assertIn('pipeline', i)
            self.assertIn('accuracy', i)
            self.assertIn('f1', i)
            self.assertIn('precision', i)
            self.assertIn('recall', i)
            self.assertIn('num_training_samples', i)
            self.assertIn('best_params', i)
            self.assertIn('cm_00', i)
            self.assertIn('cm_01', i)
            self.assertIn('cm_10', i)
            self.assertIn('cm_11', i)
