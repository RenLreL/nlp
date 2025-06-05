"""
Copyright 2025
Authors: Laélia Chi <lae.chi.22@heilbronn.dhbw.de>;
    Marco Diepold <mar.diepold.22@heilbronn.dhbw.de>;
    Yaren Sude Erol <yar.erol.22@heilbronn.dhbw.de>;
    Leon Gerke <leo.gerke.22@heilbronn.dhbw.de>;
    Dominic von Olnhausen <dom.vonolnhausen.22@heilbronn.dhbw.de>
(Edited 2025-05-26: Marco Diepold <mar.diepold.22@heilbronn.dhbw.de>)
"""


import pandas as pd
from rapidfuzz import process
from nltk.metrics.distance import edit_distance


class OutletUrlMatchmaker:


    def __init__(self, url_dataset, name_dataset):
        self.match_df = self.match_datasets(
            url_dataset,
            name_dataset
        )


    def match_datasets(self, url_dataset, name_dataset):
        threshold = 85
        choices = name_dataset["name_modification"].to_list()
        extract = lambda query: process.extractOne(
            query,
            choices,
            score_cutoff=threshold  
        )
        matches = [extract(q) for q in url_dataset["url_extract"]]

        best_match, edits = zip(
            *[
                (m[0] if m else None, m[1] if m else None) for m in matches
            ]
        )

        match_df = url_dataset.copy()
        match_df["match"] = best_match
        match_df["edits"] = edits

        match_df = match_df.merge(
            name_dataset,
            left_on="match",
            right_on="name_modification",
            how="inner"
        )
        match_df = match_df.drop_duplicates(["url"])

        return match_df
    

    def merge_datasets(self, rating_name_dataset, article_url_dataset):
        match_df = self.match_df

        articles_names = article_url_dataset.merge(
            match_df,
            left_on="source_domain",
            right_on="url",
            how="inner"
        )

        articles_ratings = articles_names.merge(
            rating_name_dataset,
            on="name",
            how="inner"
        )

        articles_ratings = articles_ratings.drop_duplicates(subset="title", keep='first', ignore_index=True)
        articles_ratings = articles_ratings[["name", "bias", "source_domain", "title", "description", "maintext"]]
        
        return articles_ratings