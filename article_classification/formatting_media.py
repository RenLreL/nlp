import pandas as pd
import ast

import sys
from pathlib import Path

# Path standardising
current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

class MediumFormatter:
    """
    A class to load, process, and format article probability data.

    Steps include:
    1. Loading classified articles from a CSV.
    2. Dropping specified columns.
    3. Converting string-formatted dictionaries in 'probabilities' to actual dictionaries.
    4. Grouping by 'Medium' and aggregating 'probabilities' into lists of dictionaries.
    5. Calculating the mean for each key within those lists of dictionaries.
    6. Sorting the keys of the resulting mean dictionaries based on a predefined order.
    7. Converting these sorted mean dictionaries into lists of their values.
    """

    def __init__(self, *, csv_path: str,):
        """
        Initializes the MediumFormatter.

        Args:
            csv_path (str): Path to the classified articles CSV file.
            columns_to_drop (list, optional): List of column names to drop.
                                               Defaults to ['Titel', 'clickable_title', 'en', 'Class'].
            classes_order (list, optional): Desired order for dictionary keys/probability classes.
                                            Defaults to ["left", "left-center", "center", "right-center", "right"].
        """
        self.csv_path = csv_path
        self.columns_to_drop = ['Titel', 'clickable_title', 'en', 'Class']
        self.classes_order = ["left", "left-center", "center", "right-center", "right"]
        self.df = None

    def _load_data(self) -> pd.DataFrame:
        """
        Loads the CSV file and drops specified columns.

        Returns:
            pd.DataFrame: The loaded and pre-processed DataFrame.
        """
        try:
            df = pd.read_csv(self.csv_path, sep=";")
            # Drop columns if they exist in the DataFrame
            cols_exist = [col for col in self.columns_to_drop if col in df.columns]
            if cols_exist:
                df = df.drop(columns=cols_exist, axis=1)
            else:
                print(f"Warning: None of the columns to drop ({self.columns_to_drop}) were found.")
            return df
        except FileNotFoundError:
            print(f"Error: CSV file not found at {self.csv_path}")
            return pd.DataFrame() # Return empty DataFrame on error
        except Exception as e:
            print(f"Error loading or preprocessing data: {e}")
            return pd.DataFrame()

    def _parse_literal_string(self, s):
        """
        Converts a string representation of a literal (like a dictionary) into a Python object.
        Handles errors by returning None.
        """
        if pd.isna(s):
            return None
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError, TypeError):
            print(f"Warning: Could not parse string '{s}' as dictionary. Returning None.")
            return None

    def _mean_of_dictionaries_in_list(self, list_of_dicts: list) -> dict:
        """
        Takes a list of dictionaries and returns a single dictionary
        where each key's value is the mean of its values across all dictionaries in the list.
        """
        # Filter out any non-dictionary elements (e.g., None, NaN) that might be in the list
        clean_list_of_dicts = [d for d in list_of_dicts if isinstance(d, dict)]

        if not clean_list_of_dicts:
            return {}

        try:
            temp_df = pd.DataFrame(clean_list_of_dicts)
            mean_series = temp_df.mean(numeric_only=True)

            # Check if mean_series is empty (e.g., if temp_df had no numeric columns)
            if mean_series.empty:
                return {}

            return mean_series.to_dict()
        except Exception as e:
            print(f"Error calculating mean for list of dicts: {list_of_dicts}. Error: {e}")
            return {}

    def _sort_dict_keys(self, original_dict: dict, order_list: list) -> dict:
        """
        Sorts the keys of a dictionary based on a defined order list.
        Keys not in order_list are excluded.
        """
        if not isinstance(original_dict, dict):
            return {} # Return empty dict or original_dict if not a dict

        sorted_dict = {}
        for key in order_list:
            if key in original_dict:
                sorted_dict[key] = original_dict[key]
        return sorted_dict

    def _dict_values_to_list(self, input_dict: dict) -> list:
        """
        Extracts values from a dictionary into a list, preserving insertion order.
        Handles non-dictionary inputs.
        """
        if not isinstance(input_dict, dict):
            return [] # Return empty list for non-dict entries
        return list(input_dict.values())

    def format_data(self) -> dict:
        """
        Executes the full data formatting pipeline.

        Returns:
            dict: A dictionary where keys are 'Medium' names and values are
                  lists of sorted and averaged probability values.
        """
        print("Starting article probability formatting pipeline...")

        # Step 1: Load data and drop initial columns
        self.df = self._load_data()
        if self.df.empty:
            print("Data loading failed or resulted in empty DataFrame. Aborting formatting.")
            return {}
        print(f"Loaded data with {self.df.shape[0]} rows and {self.df.shape[1]} columns.")

        # Step 2: Convert 'probabilities' column from string to dictionary objects
        if 'probabilities' in self.df.columns:
            print("Parsing 'probabilities' column (string to dict)...")
            self.df['probabilities'] = self.df['probabilities'].apply(self._parse_literal_string)
            # Drop rows where probabilities parsing failed
            initial_rows = self.df.shape[0]
            self.df.dropna(subset=['probabilities'], inplace=True)
            if self.df.shape[0] < initial_rows:
                print(f"Dropped {initial_rows - self.df.shape[0]} rows due to unparsable probabilities.")
        else:
            print("Error: 'probabilities' column not found in DataFrame. Aborting.")
            return {}


        # Step 3: Group by 'Medium' and aggregate 'probabilities' into lists of dictionaries
        print("Grouping by 'Medium' and aggregating probabilities into lists...")
        self.df = self.df.groupby('Medium')['probabilities'].agg(list).reset_index()

        # Step 4: Turn each list of dictionaries into a single dictionary of means
        print("Calculating mean probabilities for each Medium's list of dictionaries...")
        self.df['probabilities'] = self.df['probabilities'].apply(self._mean_of_dictionaries_in_list)
        # Drop rows where mean calculation resulted in empty dicts
        initial_rows = self.df.shape[0]
        self.df = self.df[self.df['probabilities'].apply(bool)].reset_index(drop=True)
        if self.df.shape[0] < initial_rows:
            print(f"Dropped {initial_rows - self.df.shape[0]} rows due to empty mean probability dictionaries.")


        # Step 5: Sort the keys of the mean dictionaries based on 'classes_order'
        print("Sorting dictionary keys according to defined order...")
        self.df['probabilities'] = self.df['probabilities'].apply(
            lambda x: self._sort_dict_keys(x, self.classes_order)
        )
        # Drop rows where sorting resulted in empty dicts (meaning no keys matched order)
        initial_rows = self.df.shape[0]
        self.df = self.df[self.df['probabilities'].apply(bool)].reset_index(drop=True)
        if self.df.shape[0] < initial_rows:
            print(f"Dropped {initial_rows - self.df.shape[0]} rows because sorted dictionaries were empty.")


        # Step 6: Convert the sorted dictionaries into lists of values
        print("Converting dictionaries to lists of values...")
        self.df['probabilities'] = self.df['probabilities'].apply(self._dict_values_to_list)

        print("Article probability formatting pipeline completed successfully.")

        return self.df


if __name__ == "__main__":
    formatter = MediumFormatter(csv_path="article_classification/classified_articles.csv")
    media_probs = formatter.format_data()

    print("\n--- Final formatted dictionary of probabilities per Medium ---")
    print(media_probs)

    full_csv_path = current_script_dir / "formatted_media.csv"
    media_probs.to_csv(full_csv_path, sep=";", index=False)