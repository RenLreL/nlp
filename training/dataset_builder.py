"""
Dataset Builder for News Article Bias Analysis.

This script loads news articles and media bias ratings, combines them,
cleans the data by removing media-specific clues, and splits the resulting
dataset into training, validation, and test sets.

Copyright 2025
Authors: Laélia Chi <lae.chi.22@heilbronn.dhbw.de>;
    Marco Diepold <mar.diepold.22@heilbronn.dhbw.de>;
    Yaren Sude Erol <yar.erol.22@heilbronn.dhbw.de>;
    Leon Gerke <leo.gerke.22@heilbronn.dhbw.de>;
    Dominic von Olnhausen <dom.vonolnhausen.22@heilbronn.dhbw.de>
(Edited 2025-06-20: Marco Diepold <mar.diepold.22@heilbronn.dhbw.de>)
"""

import json
import re
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from article_bias_data import ArticleBiasData
from news_article_data import NewsArticleData
from ratings_data import RatingsData


CURRENT_DIR = Path(__file__).parent
BASE_DIR = CURRENT_DIR.parent
DEFAULT_NEWS_ARTICLES_DIR = BASE_DIR / "data/268k_American_News_Articles/crawl_json/"
DEFAULT_RATINGS_FILE = BASE_DIR / "data/AllSides_Rankings/allsides.csv"
DEFAULT_OUTPUT_DIR = CURRENT_DIR / "dist"

SEED = 42


class DatasetBuilder:
    """
    Builds and processes a dataset of news articles with political bias labels.

    This class orchestrates the loading of raw news articles and bias ratings,
    merges them, preprocesses the text to remove identifying media clues,
    and splits the final dataset into training, validation, and test sets.

    Attributes:
        news_articles_directory (Path): Directory containing news article data.
        ratings_file (Path): Path to the CSV file with media bias ratings.
        output_dir (Path): Directory where processed datasets and labels are saved.
        dataset (dict[str, pd.DataFrame]): A dictionary containing the 'train',
            'val', and 'test' DataFrames. Populated after calling `build()`.
        label2id (dict[str, int]): A dictionary mapping bias labels to integer IDs.
            Populated after calling `build()`.
        id2label (dict[int, str]): A dictionary mapping integer IDs to bias labels.
            Populated after calling `build()`.
    """

    def __init__(
        self,
        news_articles_directory: Path = DEFAULT_NEWS_ARTICLES_DIR,
        ratings_file: Path = DEFAULT_RATINGS_FILE,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
    ):
        """
        Initializes the DatasetBuilder with specified data paths.

        Args:
            news_articles_directory (Path, optional): Directory of news articles.
                Defaults to DEFAULT_NEWS_ARTICLES_DIR.
            ratings_file (Path, optional): Path to the media ratings CSV.
                Defaults to DEFAULT_RATINGS_FILE.
            output_dir (Path, optional): Directory to save output files.
                Defaults to DEFAULT_OUTPUT_DIR.
        """
        self.news_articles_directory = news_articles_directory
        self.ratings_file = ratings_file
        self.output_dir = output_dir

        self.article_bias_data = None
        self.dataset = None
        self.label2id = None
        self.id2label = None

    def build(self):
        """
        Executes the full dataset building pipeline.

        This method coordinates the steps of loading, combining, processing,
        and splitting the data. It saves the resulting datasets and label
        mappings to the specified output directory.
        """
        print("(1/4) Loading and Combining Data...")
        combined_data = self._load_and_combine_data()

        print("(2/4) Preprocessing and Cleaning Data...")
        processed_data = self._preprocess_data(combined_data)

        print("(3/4) Splitting Dataset into Train, Validation, and Test Sets...")
        self.dataset = self._threefold_split(processed_data)

        print("(4/4) Saving Processed Datasets and Label Mappings...")
        self._save_datasets()
        self._save_labels()

        print("\nDataset building complete.")
        print(f"Files saved in: {self.output_dir}")

    def _load_and_combine_data(self) -> pd.DataFrame:
        """
        Loads, processes, and combines the news and ratings data.

        Returns:
            pd.DataFrame: A DataFrame containing articles matched with bias ratings.
        """
        news_articles_data = NewsArticleData(self.news_articles_directory)
        news_articles_data.load_and_process()

        ratings_data = RatingsData(self.ratings_file)
        ratings_data.load_data()

        self.article_bias_data = ArticleBiasData(
            news_article_data=news_articles_data,
            ratings_data=ratings_data
        )
        self.article_bias_data.combine_data()

        return self.article_bias_data.combined_dataset


    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies all preprocessing steps to the dataset.

        Args:
            df (pd.DataFrame): The combined article and rating data.

        Returns:
            pd.DataFrame: The fully preprocessed and cleaned DataFrame.
        """
        df = self._cap_media_fraction(df)
        df = self._remove_media_clues(df)
        df, self.label2id, self.id2label = self._enumerate_classes(
            df=df,
            label_col_name="bias",
        )
        return df

    def _remove_media_clues(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes names and other identifiers of media outlets from article text.

        This helps prevent the model from learning to associate a news source
        directly with a bias label, forcing it to rely on content.

        Args:
            df (pd.DataFrame): The DataFrame to be cleaned.

        Returns:
            pd.DataFrame: The DataFrame with media clues removed from the 'text' column.
        """
        media_name_df = self.article_bias_data.media_clues
        url_extract_df = self.article_bias_data.article_url_extracts

        media_names = media_name_df["name"].tolist()
        media_abbreviations = media_name_df["name_modification"].tolist()
        url_extracts = url_extract_df["url_extract"].to_list()

        all_clues = media_names + media_abbreviations + url_extracts
        unique_clues = list(dict.fromkeys(all_clues))
        clues_to_remove = [clue for clue in unique_clues if len(clue) > 2]

        regex_text = '|'.join(map(re.escape, clues_to_remove))
        word_boundary = r'[ \t.\(\)\[\]\{\}<>-]'
        
        regex_pattern = (
            rf'(?:^|{word_boundary})({regex_text})'
            rf'|({regex_text})(?={word_boundary}|$)'
        )
        pattern = re.compile(regex_pattern, re.IGNORECASE)

        df["text"] = df["title"] + " " + df["maintext"]
        df["text"] = df["text"].str.slice(0, 2048)
        df["text"] = df["text"].str.replace(pattern, '', regex=True)

        return df

    @staticmethod
    def _cap_media_fraction(
        df: pd.DataFrame,
        rating_column_name: str = "bias",
        media_column_name: str = "name",
        small_class_threshold: int = 50000,
        max_frac_ideal: float = 0.05,
        max_frac_small_class: float = 0.33,
    ) -> pd.DataFrame:
        """
        Limits the contribution of any single media outlet within a bias class.

        This prevents the dataset from being dominated by a few prolific sources,
        promoting a more diverse representation of outlets for each bias rating.

        Args:
            df (pd.DataFrame): The input DataFrame.
            rating_column_name (str): The column with bias ratings.
            media_column_name (str): The column with media outlet names.
            small_class_threshold (int): The threshold for a class to be 'small'.
            max_frac_ideal (float): Max fraction for a source in a large class.
            max_frac_small_class (float): Max fraction for a source in a small class.

        Returns:
            pd.DataFrame: The balanced DataFrame.
        """
        keep_parts = []
        for _, class_df in df.groupby(rating_column_name):
            c_len = len(class_df)
            max_frac = (
                max_frac_ideal
                if c_len > small_class_threshold
                else max_frac_small_class
            )
            cap = int(c_len * max_frac)

            for _, val_df in class_df.groupby(media_column_name):
                if len(val_df) > cap:
                    val_df = val_df.sample(n=cap, replace=False, random_state=SEED)
                keep_parts.append(val_df)

        capped_df = pd.concat(keep_parts, ignore_index=True)
        capped_df = capped_df.sample(frac=1, random_state=SEED)
        return capped_df.reset_index(drop=True)

    @staticmethod
    def _enumerate_classes(
        df: pd.DataFrame, label_col_name: str = "bias"
    ) -> tuple[pd.DataFrame, dict, dict]:
        """
        Creates integer labels for the bias classes.

        Args:
            df (pd.DataFrame): The DataFrame to process.
            label_col_name (str): The name of the column with class labels.

        Returns:
            tuple[pd.DataFrame, dict, dict]: A tuple containing the DataFrame with
            the new 'label_id' column, the label-to-ID mapping, and the
            ID-to-label mapping.
        """
        labels = sorted(df[label_col_name].unique())
        label2id = {lbl: idx for idx, lbl in enumerate(labels)}
        id2label = {idx: lbl for lbl, idx in label2id.items()}

        df["label_id"] = df["bias"].map(label2id).astype("int32")
        return df, label2id, id2label

    @staticmethod
    def _threefold_split(
        df: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        """
        Splits the DataFrame into train, validation, and test sets (70/15/15).

        The split is stratified to maintain the same class distribution
        across all sets.

        Args:
            df (pd.DataFrame): The DataFrame to be split.

        Returns:
            dict[str, pd.DataFrame]: A dictionary with 'train', 'val',
            and 'test' DataFrames.
        """
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

        return {"train": train_df, "val": val_df, "test": test_df}

    def _save_datasets(self):
        """
        Saves the train, validation, and test datasets to TSV files.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        for key, df in self.dataset.items():
            output_path = self.output_dir / f"data_{key}.tsv"
            df.to_csv(output_path, sep="\t", index=False)
            print(f"- Saved {key} dataset to {output_path}")

    def _save_labels(self):
        """
        Saves the label-to-ID and ID-to-label mappings to JSON files.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        label_dir = self.output_dir / "labels"
        label_dir.mkdir(exist_ok=True)

        l2i_path = label_dir / "label2id.json"
        with l2i_path.open("w", encoding="utf-8") as f:
            json.dump(self.label2id, f, indent=2)
        print(f"- Saved label-to-id mapping to {l2i_path}")
        
        i2l_path = label_dir / "id2label.json"
        with i2l_path.open("w", encoding="utf-8") as f:
            json.dump(self.id2label, f, indent=2)
        print(f"- Saved id-to-label mapping to {i2l_path}")


if __name__ == "__main__":
    builder = DatasetBuilder()

    builder.build()

    if builder.dataset:
        print("\n--- Training Data Sample ---")
        print(builder.dataset["train"].head())
        print("\n--- Label Mappings ---")
        print("Label to ID:", builder.label2id)
        print("ID to Label:", builder.id2label)