"""
Generates variations of media outlet names for matching and cleaning.

This module provides the OutletNameVariator class, which creates an extensive
list of alternative names, initials, and abbreviations from a base list of
media outlet names.

Copyright 2025
Authors: Laélia Chi <lae.chi.22@heilbronn.dhbw.de>;
    Marco Diepold <mar.diepold.22@heilbronn.dhbw.de>;
    Yaren Sude Erol <yar.erol.22@heilbronn.dhbw.de>;
    Leon Gerke <leo.gerke.22@heilbronn.dhbw.de>;
    Dominic von Olnhausen <dom.vonolnhausen.22@heilbronn.dhbw.de>
(Edited 2025-06-20: Marco Diepold <mar.diepold.22@heilbronn.dhbw.de>)
"""

import pandas as pd


class OutletNameVariator:
    """
    Creates various modifications of media outlet names for data processing.

    This class takes a dataset of media ratings and generates several alternative
    name formats, such as case-insensitive versions, initials, and abbreviations.
    These variations are used for matching articles to outlets and for cleaning
    article text.

    Attributes:
        ratings_data: An object containing media outlet names.
        name_variations (pd.DataFrame): A DataFrame containing various
            modifications of outlet names. Populated by `generate_variations()`.
        media_clues (pd.DataFrame): A comprehensive DataFrame of all name
            variations, including names with parenthetical info removed.
            Populated by `generate_variations()`.
    """

    def __init__(self, ratings_data):
        """
        Initializes the OutletNameVariator.

        Args:
            ratings_data: An object that has a 'media_outlet_names' attribute
                containing a DataFrame with a 'name' column.
        """
        self.ratings_data = ratings_data
        self.name_variations = None
        self.media_clues = None

    def generate_variations(self):
        """
        Runs the full pipeline to generate all name and clue variations.

        This method populates the `name_variations` and `media_clues`
        attributes on the instance.
        """
        self.name_variations = self._create_name_variations()
        self.media_clues = self._create_media_clues()

    def _create_media_clues(self) -> pd.DataFrame:
        """
        Creates a comprehensive list of media clues for data cleaning.

        This list includes the standard name variations plus names with
        parenthetical text removed.

        Returns:
            pd.DataFrame: A DataFrame containing all generated media clues.
        """
        names_without_parentheses = self._create_names_without_parentheses()
        media_clue_dfs = [self.name_variations, names_without_parentheses]

        media_clues = pd.concat(media_clue_dfs, axis=0, ignore_index=True)
        return media_clues

    def _create_name_variations(self) -> pd.DataFrame:
        """
        Generates and combines several types of outlet name modifications.

        Returns:
            pd.DataFrame: A filtered DataFrame of varied outlet names.
        """
        variation_datasets = [
            self._create_insensitive_names(),
            self._create_initials(partial=False),
            self._create_initials(partial=True),
            self._create_abbreviations(),
        ]

        all_variations = pd.concat(variation_datasets, axis=0, ignore_index=True)

        # Filter out very short, non-specific variations (e.g., "A", "US").
        is_specific = all_variations["name_modification"].str.len() > 2
        specific_variations = all_variations[is_specific]

        return specific_variations

    def _create_insensitive_names(self) -> pd.DataFrame:
        """Creates lowercase, character-stripped versions of names."""
        media_names = self.ratings_data.media_outlet_names.copy()
        return self._make_insensitive(media_names)

    def _create_initials(self, partial: bool = False) -> pd.DataFrame:
        """Creates full or partial initials from outlet names."""
        names_df = self.ratings_data.media_outlet_names.copy()
        names_df["name_modification"] = names_df["name"].apply(
            lambda row: self._generate_initials(row, partial)
        )
        return self._make_insensitive(
            names_df, "name_modification", "name_modification"
        )

    def _create_abbreviations(self) -> pd.DataFrame:
        """Extracts abbreviations from outlet names (e.g., 'BBC' from 'BBC News')."""
        names_df = self.ratings_data.media_outlet_names.copy()
        names_df["name_modification"] = names_df["name"].apply(
            lambda row: self._extract_abbreviation(row)
        )
        return self._make_insensitive(
            names_df, "name_modification", "name_modification"
        )

    def _create_names_without_parentheses(self) -> pd.DataFrame:
        """Removes any text in parentheses from the outlet names."""
        names_df = self.ratings_data.media_outlet_names.copy()
        names_df["name_modification"] = names_df["name"].apply(
            lambda row: self._remove_parentheses_info(row)
        )
        return names_df

    @staticmethod
    def _remove_parentheses_info(outlet_name: str) -> str:
        """
        Removes parenthetical text from a name string.

        Args:
            outlet_name (str): The name of the media outlet.

        Returns:
            str: The name with text in parentheses removed.
        """
        return outlet_name.split(" (")[0]

    @staticmethod
    def _extract_abbreviation(outlet_name: str) -> str:
        """
        Extracts an abbreviation if it is the first word of the name.

        This method assumes an abbreviation is a word in all uppercase.

        Args:
            outlet_name (str): The name of the media outlet.

        Returns:
            str: The extracted abbreviation, or an empty string if not found.
        """
        first_word = outlet_name.split(" ")[0]
        return first_word if first_word.isupper() else ""

    @staticmethod
    def _generate_initials(outlet_name: str, partial: bool = False) -> str:
        """
        Generates initials from a name.

        Args:
            outlet_name (str): The name of the media outlet.
            partial (bool): If True, appends the rest of the last word to the
                initials (e.g., "New York Times" -> "NYTimes").

        Returns:
            str: The generated initials.
        """
        name_components = outlet_name.split(" ")
        initials = "".join([word[0] for word in name_components if word])

        if partial and len(name_components) > 0:
            initials += name_components[-1][1:]

        return initials

    @staticmethod
    def _make_insensitive(
        dataframe: pd.DataFrame,
        input_col: str = "name",
        output_col: str = "name_modification",
    ) -> pd.DataFrame:
        """
        Creates a case-insensitive version of a column by removing special
        characters and converting to lowercase.

        Args:
            dataframe (pd.DataFrame): The DataFrame to modify.
            input_col (str): The source column to read from.
            output_col (str): The target column to write the result to.

        Returns:
            pd.DataFrame: The modified DataFrame.
        """
        dataframe[output_col] = (
            dataframe[input_col].str.replace(r"[()-+!.\s]", "", regex=True).str.lower()
        )
        return dataframe
    