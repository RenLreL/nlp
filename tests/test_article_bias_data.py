"""
Unit Tests for the ArticleBiasData class.

This test suite verifies the URL extraction logic.
"""

import unittest
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from training.article_bias_data import ArticleBiasData


class TestArticleBiasData(unittest.TestCase):
    """
    Tests the static helper methods of the ArticleBiasData class.
    """

    def test_extract_domain_from_url(self):
        """
        Tests that the key domain part is extracted from various URL formats.
        """
        urls_to_test = {
            "www.nytimes.com": "nytimes",
            "www.theguardian.com": "theguardian",
            "foxnews.com": "foxnews",
            "www.reuters.com": "reuters",
        }

        for url, expected in urls_to_test.items():
            with self.subTest(url=url):
                actual = ArticleBiasData._extract_domain_from_url(url)
                self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
