import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Path adjustment for imports
current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.backend.text_classification import Classifier
except ImportError:
    print("Warning: Could not import 'Classifier' from 'src.backend.text_classification'.")
    print("Please ensure 'src/backend/text_classification.py' exists and is correctly configured.")
    print("The classification step will be skipped if Classifier is not available.")
    Classifier = None # Set to None if import fails


class ArticleClassifier:
    """
    A class to load, preprocess, merge, classify news articles,
    and prepare the data for plotting.
    """

    def __init__(self):
        """
        Initializes the ArticleClassifier with paths to data files and the BERT model.
        """
        self.scraped_data_path = current_script_dir / Path("politik_artikel.csv")
        self.medium_data_path = current_script_dir / Path("title_link.csv")
        self.bert_model_dir = project_root / "modell_klassen_notebooks" / "bert_news_classifier"

        self.df_scraped = None
        self.df_medium = None
        self.df_merged = None
        self.classifier_instance = None

        # Try to initialize the classifier early
        if Classifier:
            print(f"Attempting to load classification model from: {self.bert_model_dir}")
            try:
                self.classifier_instance = Classifier(model_dir=self.bert_model_dir)
                print("Classifier initialized successfully.")
            except FileNotFoundError as e:
                print(f"Error initializing Classifier: {e}")
                print(f"Please ensure your model is saved at: {self.bert_model_dir}")
                self.classifier_instance = None
            except Exception as e:
                print(f"An unexpected error occurred during Classifier initialization: {e}")
                self.classifier_instance = None
        else:
            print("Classifier class not found. Skipping model initialization.")


    def _load_scraped_data(self) -> pd.DataFrame:
        """
        Loads and preprocesses the scraped article data.

        Returns:
            pd.DataFrame: Preprocessed scraped data.
        """
        if not self.scraped_data_path.exists():
            raise FileNotFoundError(f"Scraped data CSV not found at: {self.scraped_data_path}")

        df_scraped = pd.read_csv(self.scraped_data_path, sep=';')
        df_scraped = df_scraped.drop(['Unnamed: 4', 'De'], axis=1, errors='ignore')
        df_scraped = df_scraped.dropna(how='all')
        return df_scraped

    def _load_medium_data(self) -> pd.DataFrame:
        """
        Loads and preprocesses the medium (title/link) data.

        Returns:
            pd.DataFrame: Preprocessed medium data.
        """
        if not self.medium_data_path.exists():
            raise FileNotFoundError(f"Medium data CSV not found at: {self.medium_data_path}")

        df_medium = pd.read_csv(self.medium_data_path, sep=';')
        pattern = r"^(?:https?://)?(?:www\.)?"

        mapping = {
            "jungefreiheit": "Junge Freiheit",
            "nd-aktuell": "nd",
            "jungewelt": "Junge Welt",
            "tagesschau": "tagesschau",
            "jacobin": "Jacobin",
            "taz": "taz",
            "tichyseinblick": "Tichys Einblick"
        }

        df_medium['Medium'] = df_medium['link'].str.replace(pattern, '', regex=True)
        df_medium['Medium'] = df_medium['Medium'].str.split('.de', n=1, expand=True)[0]
        df_medium['Medium'] = df_medium['Medium'].map(mapping).fillna(df_medium['Medium'])

        # Create clickable titles with target='_blank' to open in a new tab
        df_medium['clickable_title'] = df_medium.apply(
            lambda row: f"<a href='{row['link']}' target='_blank'>{row['title']}</a>", axis=1
        )

        df_medium = df_medium.rename(columns={
            'title': 'Titel',
            'link': 'URL',
            'Medium': 'Medium',
            'clickable_title': 'clickable_title'
        })
        return df_medium

    def _merge_dataframes(self):
        """
        Merges the scraped and medium dataframes.
        """
        if self.df_scraped is None or self.df_medium is None:
            raise ValueError("DataFrames must be loaded before merging. Call _load_scraped_data and _load_medium_data first.")

        # Merge both dataframes on 'Titel' and 'URL'
        # Check that all rows remain
        self.df_merged = pd.merge(self.df_medium, self.df_scraped, on=['Titel', 'URL'])
        if self.df_merged.shape[0] != 35:
            print(f"Warning: Merged DataFrame shape is {self.df_merged.shape[0]}, expected 35 based on original script.")
        
        # URL no longer needed
        self.df_merged = self.df_merged.drop('URL', axis=1)

    def _classify_articles(self):
        """
        Applies text classification to the 'en' column of the merged dataframe.
        """
        if self.classifier_instance is None:
            print("Classification skipped: Classifier not initialized or available.")
            self.df_merged['Class'] = 'Unclassified'
            self.df_merged['probabilities'] = np.nan
            return

        if self.df_merged is None or 'en' not in self.df_merged.columns:
            raise ValueError("Merged DataFrame or 'en' column not found. Call _merge_dataframes first.")

        print("Starting article classification...")
        # Apply classification, handling potential errors for individual articles
        def safe_classify(text):
            try:
                # Ensure 'text' is not NaN or empty before classifying
                if pd.isna(text) or str(text).strip() == "":
                    return 'No_Text', {} # Or a more appropriate default/error
                return self.classifier_instance.classify(str(text))
            except Exception as e:
                print(f"Error classifying article: {e}. Article content: {text[:50]}...")
                return 'Classification_Error', {}

        # Apply classification and expand results into two new columns
        self.df_merged[['Class', 'probabilities']] = self.df_merged['en'].apply(safe_classify).apply(pd.Series)
        print("Article classification complete.")

    def analyze(self) -> pd.DataFrame:
        """
        Executes the full analysis pipeline.

        Returns:
            pd.DataFrame: The final processed DataFrame ready for plotting.
        """
        print("Starting news analysis pipeline...")
        try:
            self.df_scraped = self._load_scraped_data()
            print(f"Loaded scraped data from {self.scraped_data_path}.")

            self.df_medium = self._load_medium_data()
            print(f"Loaded medium data from {self.medium_data_path}.")

            self._merge_dataframes()
            print(f"Merged dataframes. Total articles: {self.df_merged.shape[0]}.")

            self._classify_articles()
            print("Articles classified (or skipped if model not available).")

            print("News analysis pipeline completed successfully.")
            return self.df_merged
        except FileNotFoundError as e:
            print(f"Analysis failed: {e}")
            return pd.DataFrame() # Return empty DataFrame on critical error
        except ValueError as e:
            print(f"Analysis failed due to data issue: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"An unexpected error occurred during analysis: {e}")
            return pd.DataFrame()
        
if __name__ == "__main__":
    analyer = ArticleClassifier()
    classified_data = analyer.analyze()

    print("\nFinal DataFrame:")
    print(classified_data)

    output_csv_path = "article_classification/classified_articles.csv"
    try:
        classified_data.to_csv(output_csv_path, sep=';', index=False, encoding='utf-8')
        print(f"\nDataFrame successfully saved to {output_csv_path}")
    except Exception as e:
        print(f"\nError saving DataFrame to CSV: {e}")

