"""
Copyright 2025
Authors: Laélia Chi <lae.chi.22@heilbronn.dhbw.de>;
    Marco Diepold <mar.diepold.22@heilbronn.dhbw.de>;
    Yaren Sude Erol <yar.erol.22@heilbronn.dhbw.de>;
    Leon Gerke <leo.gerke.22@heilbronn.dhbw.de>;
    Dominic von Olnhausen <dom.vonolnhausen.22@heilbronn.dhbw.de>
(Edited 2025-06-20: Marco Diepold <mar.diepold.22@heilbronn.dhbw.de>)
"""


import os
import re
import pandas as pd
from news_article_data import NewsArticleData
from ratings_data import RatingsData
from article_bias_data import ArticleBiasData
from sklearn.model_selection import train_test_split

import json
from pathlib import Path


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
NEWS_ARTICLES_DIRECTORY = os.path.join(BASE_DIR, "data/268k_American_News_Articles/crawl_json/")
RATINGS_DIRECTORY = os.path.join(BASE_DIR, "data/AllSides_Rankings/allsides.csv")
SEED = 42

class DatasetBuilder:
        

    def __init__(
            self,
            news_articles_directory_appendix=NEWS_ARTICLES_DIRECTORY,
            ratings_directory_appendix=RATINGS_DIRECTORY
    ):
        
        dataset = self.build_dataset(
            news_articles_directory_appendix=NEWS_ARTICLES_DIRECTORY,
            ratings_directory_appendix=RATINGS_DIRECTORY
        )

        self.dataset = dataset

        for key, value in dataset.items():
            value.to_csv(f"tsv/intermediary_data_{key}.tsv", sep="\t")

        self.get_id2label()
        self.get_label2id()


    def get_dataset(self):
        return self.dataset
    
    def get_label2id(self):
        out_file = Path(CURRENT_DIR).joinpath("labels/label2id.json")
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(self.label2id, f, indent=2)

        return self.label2id
    
    def get_id2label(self):
        out_file = Path(CURRENT_DIR).joinpath("labels/id2label.json")
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(self.id2label, f, indent=2)
        return self.id2label
    

    def build_dataset(
            self,
            news_articles_directory_appendix=NEWS_ARTICLES_DIRECTORY,
            ratings_directory_appendix=RATINGS_DIRECTORY
    ):
        
        print("(0/X) Loading News Articles ...")
        news_articles_data = NewsArticleData(NEWS_ARTICLES_DIRECTORY)

        print("(1/X) Loading Media Outlet Ratings")
        ratings_data = RatingsData(RATINGS_DIRECTORY)

        print("(2/X) Matching and Joining Ratings and Articles")
        self.article_bias_data = ArticleBiasData(news_articles_data, ratings_data)
        dataset = self.article_bias_data.get_combined_dataset()


        print("3/X) Weight Media Outlets, Remove Clues, Enumerate Classes, Train-Validation-Test-Split")

        dataset = self.cap_media_fraction(dataset)
        dataset = self.remove_media_clues(dataset)

        dataset, self.label2id, self.id2label = self.enumerate_classes(
            df=dataset,
            label_col_name="bias",
        )

        datasets = self.threefold_split(dataset)

        return datasets
    


    def remove_media_clues(self, df):

        media_name_df = self.article_bias_data.media_clues
        url_extract_df = self.article_bias_data.get_article_url_extracts()

        media_names = media_name_df["name"].tolist()
        media_abbreviations = media_name_df["name_modification"].tolist()
        url_extracts = url_extract_df["url_extract"].to_list()

        media_name_clues = media_names + media_abbreviations + url_extracts
        media_name_clues = list(dict.fromkeys(media_name_clues))
        to_remove = [clue for clue in media_name_clues if len(clue) > 2]
        
        regex_text = '|'.join(map(re.escape, to_remove))
        separators = r'[ \t.\(\)\[\]\{\}<>-]'

        regex_pattern = (
            rf'(?:^|{separators})({regex_text})'
            + rf'|'
            + rf'({regex_text})(?={separators}|$)'
        )

        pattern = re.compile(regex_pattern, re.IGNORECASE)

        df["text"] = df["title"] + " " + df["maintext"]
        df["text"] = df["text"].str.slice(0, 2048)
        df["text"] = df["text"].str.replace(pattern, '', regex=True)

        return df



    @staticmethod
    def threefold_split(df):
        train_val_df, test_df = train_test_split(
            df,
            stratify=df["label_id"],
            test_size=0.15,
            random_state=SEED,
        )

        val_fraction = 0.15 / 0.85

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_fraction,
            stratify=train_val_df["label_id"],
            random_state=SEED,
        )

        df_dict = {
            "train": train_df,
            "val": val_df,
            "test": test_df
        }

        return df_dict
    

    @staticmethod
    def enumerate_classes(df, label_col_name="bias"):
        labels = sorted(df[label_col_name].unique())            # alphabetical order for stability
        label2id = {lbl: idx for idx, lbl in enumerate(labels)}
        id2label = {idx: lbl for lbl, idx in label2id.items()}

        df["label_id"] = df["bias"].map(label2id).astype("int32")

        return df, label2id, id2label

    @staticmethod
    def cap_media_fraction(
        df: pd.DataFrame,
        rating_column_name="bias",
        media_column_name="name",
        small_class_threshold=50000,
        max_frac_ideal=0.05,
        max_frac_small_class=0.33,
        random_state=42
    ) -> pd.DataFrame:
        
        keep_parts = []

        for cls, class_df in df.groupby(rating_column_name):

            c_len = len(class_df)
            if c_len > small_class_threshold:
                max_frac = max_frac_ideal
            else:
                max_frac = max_frac_small_class

            cap = int(c_len * max_frac)
            for val, val_df in class_df.groupby(media_column_name):
                if len(val_df) > cap:
                    val_df = val_df.sample(
                        n=cap,
                        replace=False,
                        random_state=random_state
                    )
                keep_parts.append(val_df)

        capped_df = pd.concat(keep_parts, ignore_index=True)
        capped_df = capped_df.sample(frac=1, random_state=random_state)
        capped_df = capped_df.reset_index(drop=True)

        return capped_df


if __name__ == "__main__":
    
    dataset_builder = DatasetBuilder()