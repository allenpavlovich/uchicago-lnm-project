import unittest
import pandas as pd
from io import StringIO
from src.data.load_raw_data import load_sentiment140_raw

class TestLoadSentiment140Raw(unittest.TestCase):

    def setUp(self):
        # Create a mock CSV file
        self.mock_csv = StringIO(
            "0,1234567890,2020-01-01 00:00:00,NO_QUERY,user1,This is a negative tweet\n"
            "4,1234567891,2020-01-01 00:00:01,NO_QUERY,user2,This is a positive tweet\n"
        )
        self.mock_file_path = 'mock_data.csv'
        with open(self.mock_file_path, 'w') as f:
            f.write(self.mock_csv.getvalue())

    def tearDown(self):
        import os
        os.remove(self.mock_file_path)

    def test_load_sentiment140_raw(self):
        df = load_sentiment140_raw(self.mock_file_path)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2)
        self.assertListEqual(df.columns.tolist(), ['text', 'label'])
        self.assertEqual(df.iloc[0]['label'], 0)
        self.assertEqual(df.iloc[1]['label'], 1)

if __name__ == '__main__':
    unittest.main()