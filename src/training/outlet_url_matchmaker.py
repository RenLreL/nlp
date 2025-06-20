"""
Matches and merges news articles with media ratings using fuzzy URL matching.

This module contains the OutletUrlMatchmaker class, which links articles to
media outlets by fuzzy matching a key part of the article's URL with a list
of known media outlet name variations.

Copyright 2025
Authors: Laélia Chi <lae.chi.22@heilbronn.dhbw.de>;
    Marco Diepold <mar.diepold.22@heilbronn.dhbw.de>;
    Yaren Sude Erol <yar.erol.22@heilbronn.dhbw.de>;
    Leon Gerke <leo.gerke.22@heilbronn.dhbw.de>;
    Dominic von Olnhausen <dom.vonolnhausen.22@heilbronn.dhbw.de>
(Edited 2025-06-20: Marco Diepold <mar.diepold.22@heilbronn.dhbw.de>)
"""

import pandas as pd
from rapidfuzz import process


class OutletUrlMatchmaker:
    """
    Finds matches between article URLs and outlet names to merge datasets.

    This class uses fuzzy string matching to create a map between article URLs
    and canonical outlet names. It then uses this map to merge a full article
    dataset with a media ratings dataset.

    Attributes:
        url_df (pd.DataFrame): DataFrame with extracted URL components.
        name_variations_df (pd.DataFrame): DataFrame with variations of outlet names.
    """

    def __init__(self, url_dataset: pd.DataFrame, name_dataset: pd.DataFrame):
        """
        Initializes the OutletUrlMatchmaker with data needed for matching.

        Args:
            url_dataset (pd.DataFrame): A DataFrame containing at least a
                'url' and 'url_extract' column.
            name_dataset (pd.DataFrame): A DataFrame containing at least a
                'name' and 'name_modification' column.
        """
        self.url_df = url_dataset
        self.name_variations_df = name_dataset

    def merge_datasets(
        self, rating_name_dataset: pd.DataFrame, article_url_dataset: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merges the ratings and articles datasets using a URL-to-name map.

        This is the main public method that orchestrates the entire process of
        matching and merging the final datasets.

        Args:
            rating_name_dataset (pd.DataFrame): The full dataset of media ratings,
                containing a 'name' column.
            article_url_dataset (pd.DataFrame): The full dataset of articles,
                containing 'source_domain', 'title', and 'maintext' columns.

        Returns:
            pd.DataFrame: A clean, merged DataFrame with articles linked to their
            bias ratings.
        """
        url_to_name_map = self._create_url_to_name_map()

        articles_with_names = article_url_dataset.merge(
            url_to_name_map, left_on="source_domain", right_on="url", how="inner"
        )

        articles_with_ratings = articles_with_names.merge(
            rating_name_dataset, on="name", how="inner"
        )

        articles_with_ratings = articles_with_ratings.drop_duplicates(
            subset="title", keep="first", ignore_index=True
        )
        final_columns = ["name", "bias", "source_domain", "title", "maintext"]
        return articles_with_ratings[final_columns]

    def _create_url_to_name_map(self, threshold: int = 85) -> pd.DataFrame:
        """
        Creates a map between article URLs and canonical outlet names.

        This method uses fuzzy string matching to find the best outlet name
        variation for each extracted URL component.

        Args:
            threshold (int): The minimum similarity score (0-100) required
                for a match.

        Returns:
            pd.DataFrame: A DataFrame mapping a URL to a canonical outlet name.
        """
        choices = self.name_variations_df["name_modification"].to_list()
        queries = self.url_df["url_extract"]

        extract_one = lambda query: process.extractOne(
            query, choices, score_cutoff=threshold
        )
        matches = queries.apply(extract_one)

        match_results = matches.apply(
            lambda m: (m[0], m[1]) if m is not None else (None, None)
        )
        match_df = self.url_df.copy()
        match_df[["match", "match_score"]] = pd.DataFrame(
            match_results.to_list(), index=match_df.index
        )

        url_name_map = match_df.merge(
            self.name_variations_df,
            left_on="match",
            right_on="name_modification",
            how="inner",
        )

        url_name_map = url_name_map.drop_duplicates(["url"])
        return url_name_map[["url", "name"]]