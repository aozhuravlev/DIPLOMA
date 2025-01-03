import polars as pl

from functions.wrappers import print_status_and_time


@print_status_and_time
def load_data():
    """
    Loads and processes the training data from Parquet files.

    This function performs the following steps:
    1. Reads Parquet files from the specified file path.
    2. Converts the raw data into a Polars DataFrame.
    3. Adjusts column data types:
        - Casts the `id` column to `Int32`.
        - Casts all other columns (excluding `id`) to `Int8`.
        - Casts specific boolean indicator columns to `Boolean`.

    Returns:
        pl.DataFrame: A Polars DataFrame containing the processed data.

    Note:
        - Ensure the file path (`../data/train_data/*.pq`) points to the
        correct location of the Parquet files.
        - The `bool_cols` list specifies which columns should be converted
        to boolean type.
    """

    file_path = f"../data/train_data/*.pq"

    data_to_read = pl.read_parquet(file_path)
    df = pl.DataFrame(data_to_read)

    bool_cols = [
        "is_zero_util",
        "is_zero_over2limit",
        "is_zero_maxover2limit",
        "pclose_flag",
        "fclose_flag",
    ]
    df = df.with_columns(
        pl.col("id").cast(pl.Int32),
    )
    df = df.with_columns(pl.col(col).cast(pl.Int8) for col in df.columns if col != "id")
    df = df.with_columns(
        (pl.col(col).cast(pl.Boolean) for col in df.columns if col in bool_cols)
    )
    return df


@print_status_and_time
def load_target():
    """
    Load the target variable and merge it with the input DataFrame.

    Steps:
    1. Clone the input DataFrame to avoid modifying the original.
    2. Read the target data from `../data/train_target.csv`.
    3. Cast `id` to `Int32` and `flag` to `Int8` in the target data.
    4. Merge the target data with the input DataFrame on `id` using a left join.

    Args:
        input_df (pl.DataFrame): Input DataFrame to merge the target data with.

    Returns:
        pl.DataFrame: DataFrame with the target variable added.

    Note:
        Ensure the file path points to the correct location.
        The target CSV should have `id` and `flag` columns.
    """

    target = pl.read_csv("../data/train_target.csv")
    target = target.with_columns(
        pl.col("id").cast(pl.Int32), pl.col("flag").cast(pl.Int8)
    )

    return target['flag']
