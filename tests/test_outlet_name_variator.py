"""
Unit Tests for the OutletNameVariator class.

This test suite verifies that the string manipulation and name variation
methods in the OutletNameVariator class function as expected.
"""

import unittest
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from training.outlet_name_variator import OutletNameVariator


class TestOutletNameVariator(unittest.TestCase):
    """
    Tests the static helper methods of the OutletNameVariator class.
    """

    def test_generate_initials_multi_word(self):
        """
        Tests that initials are correctly generated from a multi-word name.
        """
        input_name = "New York Times"
        expected_initials = "NYT"
        actual_initials = OutletNameVariator._generate_initials(
            input_name, partial=False
        )
        self.assertEqual(actual_initials, expected_initials)

    def test_generate_initials_partial(self):
        """
        Tests that partial initials are correctly generated.
        """
        input_name = "New York Times"
        expected_partial = "NYTimes"
        actual_partial = OutletNameVariator._generate_initials(
            input_name, partial=True
        )
        self.assertEqual(actual_partial, expected_partial)

    def test_extract_abbreviation_present(self):
        """
        Tests that an abbreviation is correctly extracted when present.
        """
        input_name = "BBC News"
        expected_abbreviation = "BBC"
        actual_abbreviation = OutletNameVariator._extract_abbreviation(input_name)
        self.assertEqual(actual_abbreviation, expected_abbreviation)

    def test_extract_abbreviation_absent(self):
        """
        Tests that an empty string is returned when no abbreviation is present.
        """
        input_name = "The Guardian"
        expected_abbreviation = ""
        actual_abbreviation = OutletNameVariator._extract_abbreviation(input_name)
        self.assertEqual(actual_abbreviation, expected_abbreviation)

    def test_remove_parentheses_info(self):
        """
        Tests that text within parentheses is correctly removed.
        """
        input_name = "The Post (Daily)"
        expected_output = "The Post"
        actual_output = OutletNameVariator._remove_parentheses_info(input_name)
        self.assertEqual(actual_output, expected_output)

    def test_remove_parentheses_info_no_parentheses(self):
        """
        Tests that a name without parentheses remains unchanged.
        """
        input_name = "The New York Times"
        expected_output = "The New York Times"
        actual_output = OutletNameVariator._remove_parentheses_info(input_name)
        self.assertEqual(actual_output, expected_output)


if __name__ == "__main__":
    unittest.main()
