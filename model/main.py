"""
Copyright 2025
Authors: Laélia Chi <lae.chi.22@heilbronn.dhbw.de>;
    Marco Diepold <mar.diepold.22@heilbronn.dhbw.de>;
    Yaren Sude Erol <yar.erol.22@heilbronn.dhbw.de>;
    Leon Gerke <leo.gerke.22@heilbronn.dhbw.de>;
    Dominic von Olnhausen <dom.vonolnhausen.22@heilbronn.dhbw.de>
(Edited 2025-05-26: Marco Diepold <mar.diepold.22@heilbronn.dhbw.de>)
"""


import os
import pandas as pd
from news_article_data import NewsArticleData
from ratings_data import RatingsData
from article_bias_data import ArticleBiasData


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
NEWS_ARTICLES_DIRECTORY = os.path.join(BASE_DIR, "data/268k_American_News_Articles/crawl_json/")
RATINGS_DIRECTORY = os.path.join(BASE_DIR, "data/AllSides_Rankings/allsides.csv")


if __name__ == "__main__":
    print("(0/X) Loading News Articles ...")
    news_articles_data = NewsArticleData(NEWS_ARTICLES_DIRECTORY)

    print("(1/X) Loading Media Outlet Ratings")
    ratings_data = RatingsData(RATINGS_DIRECTORY)

    print("(2/X) Matching and Joining Ratings and Articles")
    article_bias_data = ArticleBiasData(news_articles_data, ratings_data)
    dataset = article_bias_data.get_combined_dataset()
    dataset.to_csv("intermediary_data.tsv", sep="\t") #remove export after succesful implementation
