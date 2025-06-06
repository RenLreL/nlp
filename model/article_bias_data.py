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
from outlet_name_variator import OutletNameVariator
from outlet_url_matchmaker import OutletUrlMatchmaker

class ArticleBiasData:


    def __init__(self, news_article_data, ratings_data):
        self.news_articles = news_article_data
        self.ratings = ratings_data
        self.article_url_extracts = self.get_article_url_extracts()
        outlet_name_variator = OutletNameVariator(self.ratings)
        self.outlet_name_variations = outlet_name_variator.get_outlet_name_variations()
        self.media_clues = outlet_name_variator.get_media_clue_variations()

        outlet_url_matchmaker = OutletUrlMatchmaker(
            self.article_url_extracts,
            self.outlet_name_variations
        )

        self.combined_dataset = outlet_url_matchmaker.merge_datasets(
            rating_name_dataset=self.ratings.dataset,
            article_url_dataset=self.news_articles.dataset
        )


    def get_combined_dataset(self):

        return self.combined_dataset


    def get_article_url_extracts(self):
        url_list = self.news_articles.media_outlet_urls
        url_extracts = [self.url_extract(url) for url in url_list]

        url_df = pd.DataFrame(
            {
                "url": url_list,
                "url_extract": url_extracts
            }
        )

        return url_df
    

    @staticmethod
    def url_extract(full_url):
        words = full_url.split(".")
        word_length = len(words)
        cutoff_a = word_length-3
        cutoff_b = word_length-1
        if cutoff_a == 0:
            cutoff_a = 1
        extractions = words[cutoff_a:cutoff_b]
        name = ''.join(extractions)

        return name
    


