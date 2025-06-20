"""
Loads and processes news article data from a directory of JSON files.

This module defines the NewsArticleData class, which is responsible for
reading raw news articles from JSON files, cleaning the text content, and
storing the results in a pandas DataFrame.

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


class NewsArticleData:
    """
    Loads, cleans, and manages news article data from a source directory.

    This class handles the entire pipeline of reading multiple JSON files,
    combining them into a single DataFrame, and performing basic text cleaning.

    Attributes:
        data_path (Path): The path to the directory containing the news article
            JSON files.
        dataset (pd.DataFrame): The processed DataFrame containing the news
            articles. This is populated after `load_and_process()` is called.
        media_outlet_urls (np.ndarray): An array of unique source domains
            found in the dataset. Populated after `load_and_process()` is called.
    """

    def __init__(self, news_article_data_path: Path):
        """
        Initializes the NewsArticleData class with the path to the data.

        Args:
            news_article_data_path (Path): The directory path containing JSON files.
        """
        self.data_path = news_article_data_path
        self.dataset = None
        self.media_outlet_urls = None

    def load_and_process(self):
        """
        Executes the full data loading and cleaning pipeline.

        This method reads the source files, cleans the data, and populates
        the `dataset` and `media_outlet_urls` attributes.
        """
        raw_articles = self._read_news_articles()
        cleaned_articles = self._clean_news_articles(raw_articles)
        self.dataset = cleaned_articles
        self.media_outlet_urls = cleaned_articles["source_domain"].unique()

    def _read_news_articles(self) -> pd.DataFrame:
        """
        Reads, combines, and structures news articles from JSON files.

        The method performs the following steps:
        1. Finds all JSON files in the specified directory.
        2. Reads each JSON file into a pandas DataFrame.
        3. Concatenates the DataFrames, which results in a transposed view.
        4. Transposes the combined DataFrame back to the correct orientation.
        5. Removes duplicate articles based on the 'title' column.

        Returns:
            pd.DataFrame: A DataFrame containing the combined and cleaned
            news articles.
        """
        json_files = list(self.data_path.glob("*.json"))
        df_list = [pd.read_json(filepath) for filepath in json_files]

        combined_df = pd.concat(df_list, axis=1)

        transposed_df = combined_df.T
        transposed_df.columns = list(combined_df.index)

        unique_articles = transposed_df.drop_duplicates(
            subset="title", keep="first"
        )
        return unique_articles.copy()

    @staticmethod
    def _clean_news_articles(news_articles: pd.DataFrame) -> pd.DataFrame:
        """
        Performs basic cleaning on the text columns of the articles DataFrame.

        This method casts key columns to string type and removes unwanted
        characters and tabs using regular expressions.

        Args:
            news_articles (pd.DataFrame): The DataFrame to be cleaned.

        Returns:
            pd.DataFrame: The cleaned DataFrame.
        """
        text_columns = ["title", "description", "maintext"]

        for col in text_columns:
            news_articles[col] = news_articles[col].astype(str)
            news_articles[col] = news_articles[col].str.replace(
                r"[^a-zA-Z0-9 .,;:!?(){}\"%-]+", "", regex=True
            )
            news_articles[col] = news_articles[col].str.replace(r"\t", "", regex=True)

        return news_articles