import pandas as pd

import sys
from pathlib import Path

# Path standardising
current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

class ArticleFormatter:
    """
    A class to further format and prepare news article data for display or visualization.
    It takes a DataFrame, typically processed by a ArticleClassifier, and applies final
    formatting steps such as adding HTML breaks to titles and sorting.
    """

    def __init__(self, *, classified_df: pd.DataFrame):
        """
        Initializes the ArticleFormatter with a classified DataFrame and formatting parameters.
        """
        if not isinstance(classified_df, pd.DataFrame):
            raise TypeError("Input 'classified_df' must be a pandas DataFrame.")

        # Work on a copy of the DataFrame to avoid modifying the original
        self.df = classified_df.copy()
        self.articles_per_medium = 5

        self.classes_order = ["left", "left-center", "center", "right-center", "right"]

    def _format_titles(self, titles_list: list) -> str:
        """
        Adds <br> before every string in the list except the first one and joins them.
        This is used for displaying multiple titles within a single cell
        in a web view.

        Args:
            titles_list (list): A list of strings, where each string is a clickable title.

        Returns:
            str: A single string with titles separated by '<br>' tags.
        """
        if not titles_list:
            return "" # Return empty string for empty lists

        formatted_parts = []
        # Add the first title as is, ensuring it's a string
        formatted_parts.append(str(titles_list[0]))

        # Add <br> before subsequent titles, ensuring each is a string
        for title in titles_list[1:]:
            formatted_parts.append(f"<br>{str(title)}")

        return "".join(formatted_parts)

    def format_data(self) -> pd.DataFrame:
        """
        Applies all specified formatting and processing steps to the DataFrame.
        This includes:
        1. Dropping redundant columns ('Titel', 'en').
        2. Re-grouping data and calculating percentages.
        3. Applying HTML break tags to combine clickable titles within a cell.
        4. Sorting the DataFrame by 'Medium' and a defined 'Class' order.

        Returns:
            pd.DataFrame: The final formatted DataFrame.
        """
        print("Starting data formatting pipeline...")

        # Drop the columns that are no longer needed
        columns_to_drop = [col for col in ['Titel', 'en'] if col in self.df.columns]
        if columns_to_drop:
            self.df = self.df.drop(columns=columns_to_drop)
            print(f"Dropped columns: {', '.join(columns_to_drop)}")

        # Combine the clickable titles to have one row for each Medium-Class
        # This check prevents re-grouping if 'clickable_title' is already a list of titles
        if not self.df.empty and (self.df['clickable_title'].dtype == 'object' and
                                  not isinstance(self.df.loc[self.df.index[0], 'clickable_title'], list)):
             print("Re-grouping clickable titles...")
             self.df = self.df.groupby(['Medium', 'Class'])['clickable_title'].agg(list).reset_index()

        # Calculate the percentage of each Class
        print("Calculating 'Value' and 'Percentage'...")
        self.df['Value'] = self.df['clickable_title'].apply(lambda x: (len(x) / self.articles_per_medium) * 100)
        self.df['Percentage'] = self.df['Value'].apply(lambda x: f"{x:.1f}%")


        # Apply the function to format 'clickable_title' with <br> tags
        print("Applying HTML break formatting to clickable titles...")
        # Ensure that 'clickable_title' is a list before applying _format_titles
        # The previous groupby should ensure this, but an extra check is safe.
        if not self.df.empty and not isinstance(self.df.loc[self.df.index[0], 'clickable_title'], list):
            # If for some reason it's not a list, convert it to a list containing itself
            self.df['clickable_title'] = self.df['clickable_title'].apply(lambda x: [x] if pd.notna(x) else [])

        self.df['clickable_title'] = self.df['clickable_title'].apply(self._format_titles)


        # Sort the DataFrame based on 'Medium' and the defined 'Class' order
        print("Sorting DataFrame by Medium and Class order...")
        if 'Class' in self.df.columns and self.classes_order:
            # Ensure 'Class' column is of Categorical type for custom sorting
            self.df['Class'] = pd.Categorical(self.df['Class'], categories=self.classes_order, ordered=True)
            self.df = self.df.sort_values(by=['Medium', 'Class'])
        else:
            print("Warning: Cannot sort by 'Class'. 'Class' column not found or classes_order is empty.")

        # Reset the DataFrame index after sorting
        self.df = self.df.reset_index(drop=True)

        print("Data formatting pipeline completed successfully.")
        return self.df

# --- Example Usage (for demonstration, typically would be in a separate main script) ---
if __name__ == "__main__":
    classified_data = pd.read_csv("article_classification/classified_articles.csv", sep=";")
    
    data_formatter = ArticleFormatter(classified_df=classified_data)
    formatted_df = data_formatter.format_data()

    print("\nFinal formatted DataFrame:")
    print(formatted_df)

    output_csv_path = "article_classification/formatted_articles.csv"
    try:
        formatted_df.to_csv(output_csv_path, sep=';', index=False, encoding='utf-8')
        print(f"\nDataFrame successfully saved to {output_csv_path}")
    except Exception as e:
        print(f"\nError saving DataFrame to CSV: {e}")
