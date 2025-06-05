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
import os


class NewsArticleData:
    

    def __init__(self, news_article_data_path):
        self.data_path = news_article_data_path
        self.dataset = self.get_news_articles()
        self.media_outlet_urls = self.get_media_outlet_urls()


    def get_news_articles(self):
        news_articles = self.read_news_articles()
        news_articles = NewsArticleData.clean_news_articles(news_articles)
        return news_articles


    def read_news_articles(self):
        json_dir = self.data_path
        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        json_filepaths = [os.path.join(json_dir, file) for file in json_files]
        df_list = [pd.read_json(filepath) for filepath in json_filepaths]

        # df_list = [pd.read_json(os.path.join(json_dir, file)) for file in json_files]
        combined_df = pd.concat(df_list, axis=1)
        column_names = list(combined_df.index)

        transposed_df = combined_df.T
        transposed_df.columns = column_names

        news_articles = transposed_df.drop_duplicates(subset="title", keep='first')
        
        return news_articles
    

    def get_media_outlet_urls(self):
        outlet_urls = self.dataset["source_domain"].unique()

        return outlet_urls
    

    @staticmethod
    def clean_news_articles(news_articles):

        format_columns = [
            "title",
            "description",
            "maintext"
        ]

        for c in format_columns:
            news_articles[c] = news_articles[c].astype(str)
            news_articles[c] = news_articles[c].str.replace(r'[^a-zA-Z0-9 .,;:!?(){}"%-]+', '', regex=True)
            news_articles[c] = news_articles[c].str.replace(r'\t', '', regex=True)

        return news_articles
