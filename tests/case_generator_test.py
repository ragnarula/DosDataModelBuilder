import unittest
from kaizen import AscendingSizeCaseGenerator
import pandas as pd


class AscendingSizeCaseGeneratorTest(unittest.TestCase):

    test_data = pd.DataFrame.from_dict({
        'x': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        'Class': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    })

    def test_init_throws_value_error(self):
        """
        The constructor should throw a value error when test and train sets overlap
        :return: 
        """

        def fixture():
            AscendingSizeCaseGenerator(self.test_data, 'Class', upper=0.8)
        self.assertRaises(ValueError, fixture)

    def test_get_iterator_returns_correct_format(self):
        """
        The get_iterator method should return a tuple of 4 lists of ints
        :return: 
        """

        fixture = AscendingSizeCaseGenerator(self.test_data, 'Class')
        iter = fixture.get_iterator()

        for i in iter:
            train_samples, test_samples = i
            self.assertEqual(len(test_samples), 3)
            for s in train_samples:
                self.assertNotIn(s, test_samples)
