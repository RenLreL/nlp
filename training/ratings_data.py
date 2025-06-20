"""
Loads media bias rating data from a CSV file.

This module provides the RatingsData class, which is responsible for loading
a dataset of media outlet bias ratings and extracting a unique list of
outlet names.

Copyright 2025
Authors: Laélia Chi <lae.chi.22@heilbronn.dhbw.de>;
    Marco Diepold <mar.diepold.22@heilbronn.dhbw.de>;
    Yaren Sude Erol <yar.erol.22@heilbronn.dhbw.de>;
    Leon Gerke <leo.gerke.22@heilbronn.dhbw.de>;
    Dominic von Olnhausen <dom.vonolnhausen.22@heilbronn.dhbw.de>
(Edited 2025-06-20: Marco Diepold <mar.diepold.22@heilbronn.dhbw.de>)
"""

import pandas as pd
from pathlib import Path


class RatingsData:
    """
    Loads and prepares media outlet bias ratings from a CSV file.

    Attributes:
        data_path (Path): The file path to the ratings CSV data.
        dataset (pd.DataFrame): The full DataFrame of ratings data.
            Populated after `load_data()` is called.
        media_outlet_names (pd.DataFrame): A DataFrame containing a single
            'name' column with unique media outlet names. Populated after
            `load_data()` is called.
    """

    def __init__(self, ratings_data_path: Path):
        """
        Initializes the RatingsData class.

        Args:
            ratings_data_path (Path): The path to the ratings CSV file.
        """
        self.data_path = ratings_data_path
        self.dataset = None
        self.media_outlet_names = None

    def load_data(self):
        """
        Loads data from the CSV file and extracts unique media outlet names.

        This method reads the data from the path provided at initialization.
        It populates the `dataset` attribute with the full ratings table
        and the `media_outlet_names` attribute with a DataFrame of unique names.
        """
        self.dataset = pd.read_csv(self.data_path)
        unique_names_list = self.dataset["name"].unique()
        self.media_outlet_names = pd.DataFrame({"name": unique_names_list})