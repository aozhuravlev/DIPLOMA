import polars as pl
import pandas as pd
import numpy as np

from typing import Set

from feature_engineering_pl import print_status_and_time

def drop_constant_features(input_df: pl.DataFrame) -> pl.DataFrame:
    df = input_df.clone()
    non_constant_cols = [
        col for col in df.columns 
        if df[col].n_unique() > 1
    ]
    return df.select(non_constant_cols)


@print_status_and_time
def remove_hi_corr_feats(
    input_df: pl.DataFrame, threshold: float = 0.9
) -> pl.DataFrame:
    """
    Remove features with high correlation from the input DataFrame.

    This function computes the correlation matrix of the input DataFrame and removes
    features that are highly correlated with other features, based on a specified
    correlation threshold. The features with an absolute correlation greater than
    the threshold are dropped.

    Args:
        input_df (pl.DataFrame): The input Polars DataFrame containing the features.
        threshold (float, optional): The correlation threshold above which features
                                     are considered highly correlated. Default is 0.9.

    Returns:
        pl.DataFrame: The input DataFrame with highly correlated features removed.

    Note:
        - The function uses the upper triangle of the correlation matrix to identify
          correlated features, ensuring that each pair is only considered once.
        - By default, features with a correlation greater than 0.9 will be removed.
    """
    df = input_df.clone()

    # Compute correlation matrix
    corr_matrix = df.corr().to_numpy()

    # Extract the upper triangle of the correlation matrix (excluding diagonal)
    upper_tri = np.triu(corr_matrix, k=1)

    to_drop: Set[str] = set()

    # Identify features to drop based on the correlation threshold
    for i in range(upper_tri.shape[0]):
        for j in range(i + 1, upper_tri.shape[1]):
            if abs(upper_tri[i, j]) > threshold:
                to_drop.add(df.columns[j])

    # Drop the highly correlated features
    return df.drop(list(to_drop))


@print_status_and_time
def convert_to_pandas(input_df: pl.DataFrame) -> pd.DataFrame:
    """
    Convert a Polars DataFrame to a Pandas DataFrame.

    This function converts the input Polars DataFrame into a Pandas DataFrame
    for compatibility with libraries or methods that require Pandas.

    Args:
        input_df (pl.DataFrame): The input Polars DataFrame to be converted.

    Returns:
        pd.DataFrame: The corresponding Pandas DataFrame.

    Note:
        - The conversion is performed using the `to_pandas` method provided by Polars.
    """
    return input_df.to_pandas()
