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
        total_size = 284807
        X_train, X_test, y_train, y_test = retrieve_data()
        self.assertEqual(total_size,len(X_train)+len(X_test))
        self.assertEqual(total_size,len(y_train)+len(y_test))
        self.assertEqual(int(total_size*0.67), len(y_train))
        self.assertEqual(int(total_size*0.67), len(X_train))
        self.assertEqual(int(total_size*0.33+1), len(y_test))
        self.assertEqual(int(total_size*0.33+1), len(X_test))


    def test_undersampling(self):
        """
        Checks that the ratio property of the function is working properly
        """
        with self.assertRaises(ValueError):
            retrieve_data(undersampling=True, ratio=1.1)
        X_train, X_test, y_train, y_test = retrieve_data( undersampling=True,\
                                                          ratio=0.5)
        self.assertEqual(1476,len(X_train)+len(X_test))
        self.assertEqual(1476,len(y_train)+len(y_test))
        self.assertEqual(988, len(y_train))
        self.assertEqual(988, len(X_train))
        self.assertEqual(488, len(y_test))
        self.assertEqual(488, len(y_test))



if __name__ == '__main__':
    unittest.main()
