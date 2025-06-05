"""
Copyright 2025
Authors: Laélia Chi <lae.chi.22@heilbronn.dhbw.de>;
    Marco Diepold <mar.diepold.22@heilbronn.dhbw.de>;
    Yaren Sude Erol <yar.erol.22@heilbronn.dhbw.de>;
    Leon Gerke <leo.gerke.22@heilbronn.dhbw.de>;
    Dominic von Olnhausen <dom.vonolnhausen.22@heilbronn.dhbw.de>
(Edited 2025-05-26: Marco Diepold <mar.diepold.22@heilbronn.dhbw.de>)
"""

from sklearn.utils import resample
import pandas as pd
import numpy as np

MAXIMUM_CLASS_ARTICLE_SHARE = 0.1

def balance_and_format():
    ...


def balance():
    ...






def undersample_dataframe(df, class_col_name):
    class_counts = df[class_col_name].value_counts()
    min_count = class_counts.min()
    classes = class_counts.index

    resampled_dfs = []
    for cls in classes:
        class_subset = df[df[class_col_name] == cls]
        resampled = resample(
            class_subset,
            replace=False,
            n_samples=min_count,
            random_state=42
        )
        resampled_dfs.append(resampled)

    balanced_df = pd.concat(resampled_dfs)
    balanced_df = balanced_df.sample(frac=1, random_state=42) #shuffle
    balanced_df = balanced_df.reset_index(drop=True)
    return balanced_df