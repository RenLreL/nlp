"""
Unit Tests for the DatasetBuilder class.

This test suite verifies the data processing methods like capping, splitting,
and cleaning within the DatasetBuilder class.
"""

import unittest
import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from training.dataset_builder import DatasetBuilder


class TestDatasetBuilder(unittest.TestCase):
    """
    Tests the static helper methods of the DatasetBuilder class.
    """

    def test_cap_media_fraction(self):
        """
        Tests that the contribution of media outlets is correctly capped.
        """
        data = {
            "name": ["A"] * 10 + ["B"] * 2 + ["C"] * 8,
            "bias": ["Left"] * 20,
        }
        df = pd.DataFrame(data)
        capped_df = DatasetBuilder._cap_media_fraction(
            df,
            rating_column_name="bias",
            media_column_name="name",
            small_class_threshold=30,
            max_frac_small_class=0.33,
        )
        cap = int(20 * 0.33)
        self.assertLessEqual(len(capped_df[capped_df["name"] == "A"]), cap)
        self.assertEqual(len(capped_df[capped_df["name"] == "B"]), 2)
        self.assertLessEqual(len(capped_df[capped_df["name"] == "C"]), cap)
        self.assertEqual(len(capped_df), 6 + 2 + 6)

    def test_enumerate_classes(self):
        """
        Tests that bias labels are correctly converted to integer IDs.
        """
        df = pd.DataFrame({"bias": ["Left", "Right", "Center", "Left"]})
        processed_df, label2id, id2label = DatasetBuilder._enumerate_classes(df)
        expected_label2id = {"Center": 0, "Left": 1, "Right": 2}
        expected_id2label = {0: "Center", 1: "Left", 2: "Right"}
        self.assertEqual(label2id, expected_label2id)
        self.assertEqual(id2label, expected_id2label)
        self.assertTrue("label_id" in processed_df.columns)
        self.assertEqual(processed_df["label_id"].tolist(), [1, 2, 0, 1])

    def test_threefold_split(self):
        """
        Tests that the dataset split is stratified correctly, preserving
        the relative proportions of each class in the `label_id` column.
        """
        labels = ([0] * 60) + ([1] * 30) + ([2] * 10)
        df = pd.DataFrame({
            "label_id": labels,
            "features": range(100)
        })
        original_proportions = df["label_id"].value_counts(normalize=True)
        datasets = DatasetBuilder._threefold_split(df)
        total_samples = sum(len(d) for d in datasets.values())
        self.assertEqual(total_samples, 100)
        self.assertAlmostEqual(len(datasets["test"]), 15, delta=1)
        self.assertAlmostEqual(len(datasets["val"]), 15, delta=1)
        self.assertAlmostEqual(len(datasets["train"]), 70, delta=2)

        for split_name, split_df in datasets.items():
            with self.subTest(split=split_name):
                self.assertFalse(split_df.empty, f"Split '{split_name}' should not be empty.")

                split_proportions = split_df["label_id"].value_counts(normalize=True)
                pd.testing.assert_series_equal(
                    original_proportions,
                    split_proportions,
                    check_names=False,
                    atol=0.15,
                    check_index_type=False,
                    obj=f"Proportions in '{split_name}' split"
                )


if __name__ == "__main__":
    unittest.main()
