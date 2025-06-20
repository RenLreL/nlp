"""
Unit Tests for the NewsArticleData class.

This test suite verifies the text cleaning functionality.
"""

import unittest
import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from training.news_article_data import NewsArticleData


class TestNewsArticleData(unittest.TestCase):
    """
    Tests the static helper methods of the NewsArticleData class.
    """

    def test_clean_news_articles(self):
        """
        Tests that unwanted characters are removed from text columns.
        """
        data = {
            "title": ["Title with\t tab and weird$#char"],
            "description": ["Description with @symbols"],
            "maintext": ["Main text is clean."],
        }
        df = pd.DataFrame(data)
        cleaned_df = NewsArticleData._clean_news_articles(df)
        self.assertEqual(cleaned_df["title"].iloc[0], "Title with tab and weirdchar")
        self.assertEqual(cleaned_df["description"].iloc[0], "Description with symbols")
        self.assertEqual(cleaned_df["maintext"].iloc[0], "Main text is clean.")


if __name__ == "__main__":
    unittest.main()
