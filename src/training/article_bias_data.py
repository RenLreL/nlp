"""
Prepares and merges news article data with media bias ratings.

This module defines the ArticleBiasData class, which is responsible for
taking raw news article data and media bias ratings, generating necessary
matching keys (URL extracts, outlet name variations), and merging them
into a single dataset.

Copyright 2025
Authors: Laélia Chi <lae.chi.22@heilbronn.dhbw.de>;
    Marco Diepold <mar.diepold.22@heilbronn.dhbw.de>;
    Yaren Sude Erol <yar.erol.22@heilbronn.dhbw.de>;
    Leon Gerke <leo.gerke.22@heilbronn.dhbw.de>;
    Dominic von Olnhausen <dom.vonolnhausen.22@heilbronn.dhbw.de>
(Edited 2025-06-20: Marco Diepold <mar.diepold.22@heilbronn.dhbw.de>)
"""

import pandas as pd
from .outlet_name_variator import OutletNameVariator
from .outlet_url_matchmaker import OutletUrlMatchmaker


class ArticleBiasData:
    """
    Combines news articles with media bias ratings based on URL matching.

    This class orchestrates the process of generating variations of media outlet
    names, extracting key domain information from article URLs, and then
    merging the two datasets together.

    Attributes:
        news_articles (NewsArticleData): An object containing the news articles dataset.
        ratings (RatingsData): An object containing the media bias ratings dataset.
        combined_dataset (pd.DataFrame): The final merged dataset containing
            articles and their corresponding bias ratings. Populated after
            calling `combine_data()`.
        media_clues (pd.DataFrame): A DataFrame of potential media outlet names
            and abbreviations used for data cleaning. Populated after
            calling `combine_data()`.
        article_url_extracts (pd.DataFrame): A DataFrame mapping full article
            URLs to extracted domain names. Populated after calling `combine_data()`.
    """

    def __init__(self, news_article_data, ratings_data):
        """
        Initializes the ArticleBiasData with necessary data sources.

        Args:
            news_article_data: An object with a 'dataset' attribute containing
                a DataFrame of news articles and a 'media_outlet_urls' attribute.
            ratings_data: An object with a 'dataset' attribute containing
                a DataFrame of media bias ratings.
        """
        self.news_articles = news_article_data
        self.ratings = ratings_data

        self.article_url_extracts = None
        self.outlet_name_variations = None
        self.media_clues = None
        self.combined_dataset = None

    def combine_data(self):
        """
        Executes the full pipeline to merge articles with bias ratings.

        This method generates the necessary keys for matching and then performs
        the merge, populating the instance's attributes with the results.
        """
        self.article_url_extracts = self._create_article_url_extracts()

        name_variator = OutletNameVariator(self.ratings)
        name_variator.generate_variations()
        self.outlet_name_variations = name_variator.name_variations
        self.media_clues = name_variator.media_clues

        url_matchmaker = OutletUrlMatchmaker(
            url_dataset=self.article_url_extracts,
            name_dataset=self.outlet_name_variations,
        )

        self.combined_dataset = url_matchmaker.merge_datasets(
            rating_name_dataset=self.ratings.dataset,
            article_url_dataset=self.news_articles.dataset,
        )

    def _create_article_url_extracts(self) -> pd.DataFrame:
        """
        Creates a DataFrame of URLs and their extracted domain names.

        Returns:
            pd.DataFrame: A DataFrame with 'url' and 'url_extract' columns.
        """
        url_list = self.news_articles.media_outlet_urls
        url_extracts = [self._extract_domain_from_url(url) for url in url_list]

        url_df = pd.DataFrame({"url": url_list, "url_extract": url_extracts})
        return url_df

    @staticmethod
    def _extract_domain_from_url(full_url: str) -> str:
        """
        Extracts a key part of a domain name from a full URL string.

        Note: This method uses a specific logic that is primarily effective
        for URLs of the format 'www.domain.com', where it extracts 'domain'.
        Its behavior may be less predictable for other URL structures, such as
        'domain.com' or 'domain.co.uk'. The original functionality is preserved.

        Args:
            full_url (str): The URL to process.

        Returns:
            str: The extracted domain part.
        """
        domain_parts = full_url.split(".")
        num_parts = len(domain_parts)

        start_index = num_parts - 3
        end_index = num_parts - 1

        if start_index == 0:
            start_index = 1

        if domain_parts[0] != "www":
            start_index = 0

        extractions = domain_parts[start_index:end_index]
        name = "".join(extractions)

        return name