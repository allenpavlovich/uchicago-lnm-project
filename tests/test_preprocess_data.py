import unittest
import pandas as pd
import html
from src.data.preprocess_data import clean_text, preprocess_text, preprocess_data

class TestPreprocessData(unittest.TestCase):

    def test_clean_text(self):
        # Test with a string containing URLs, mentions, hashtags, special characters, and numbers
        self.assertEqual(clean_text("Check this out! http://example.com @user #hashtag 123"), "check this out")
        
        # Test with a string that is already clean
        self.assertEqual(clean_text("This is a clean text"), "this is a clean text")

        # Test with a string containing HTML entities
        self.assertEqual(clean_text("watching &quot;House&quot; "), "watching house")

    def test_preprocess_text(self):
        # Test with a string containing stopwords and words that need lemmatization
        self.assertEqual(preprocess_text("this is a test of the preprocessing function"), "test preprocessing function")
        
        # Test with a string that is already preprocessed
        self.assertEqual(preprocess_text("test preprocessing function"), "test preprocessing function")

    def test_preprocess_data(self):
        # Test with a DataFrame containing a column with text data
        df = pd.DataFrame({'text': ["Check this out! http://example.com @user #hashtag 123", "This is a clean text"]})
        processed_df = preprocess_data(df, 'text')
        expected_df = pd.DataFrame({'text': ["Check this out! http://example.com @user #hashtag 123", "This is a clean text"],
                                    'clean_text': ["check", "clean text"]})
        
        pd.testing.assert_frame_equal(processed_df, expected_df)
        
        # Test with an empty DataFrame
        df_empty = pd.DataFrame({'text': []})
        processed_df_empty = preprocess_data(df_empty, 'text')
        expected_df_empty = pd.DataFrame({'text': [], 'clean_text': []})
        pd.testing.assert_frame_equal(processed_df_empty, expected_df_empty)

if __name__ == '__main__':
    unittest.main()