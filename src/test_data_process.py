from data_process import retrieve_data
import numpy as np
import unittest

class DataProcessor_test(unittest.TestCase):
    """
    This class is a testfunction for the dataprocessor.
    """
    def test_canary(self):
        """testing that the simplest case works."""
        self.assertEqual(2, 2)

    def test_arrays(self):
        """
        testing that the ouput has correct dimension
        """


    def test_undersampling(self):
        """
        Checks that the ratio property of the function is working properly
        """
        with self.assertRaises(ValueError):
            retrieve_data(undersampling=True, ratio=1.1)
        X_train, X_test, y_train, y_test = retrieve_data( undersampling=True,ratio=1)
        print(len(X_train))

if __name__ == '__main__':
    unittest.main()
