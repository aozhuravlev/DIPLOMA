import warnings

import polars as pl

from functions.wrappers import print_status_and_time

warnings.filterwarnings("ignore")


pre_loans_cols = [
    "pre_loans5",
    "pre_loans530",
    "pre_loans3060",
    "pre_loans6090",
    "pre_loans90",
]
is_zero_loans_upto_60 = ["is_zero_loans5", "is_zero_loans530", "is_zero_loans3060"]
is_zero_loans_over_60 = ["is_zero_loans6090", "is_zero_loans90"]
is_zero_loans_total = is_zero_loans_upto_60 + is_zero_loans_over_60
enc_paym_sum_03 = [f"enc_paym_{i}" for i in range(4)]
enc_paym_sum_424 = [f"enc_paym_{i}" for i in range(4, 25)]


@print_status_and_time
def horizontal_sum(input_df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute horizontal sums for specific columns and add the results as new columns.

    This function clones the input DataFrame and calculates horizontal sums for
    predefined groups of columns. The results are added as new columns with appropriate
    data types.

    Args:
        input_df (pl.DataFrame): Input Polars DataFrame containing the necessary columns.

    Returns:
        pl.DataFrame: A Polars DataFrame with additional columns containing the computed sums.
    """
    df = input_df.clone()
    df = df.with_columns(
        pl.sum_horizontal(pre_loans_cols).alias("pre_loans_total").cast(pl.Int8),
        pl.sum_horizontal(is_zero_loans_upto_60)
        .alias("is_zero_loans_upto_60")
        .cast(pl.Int8),
        pl.sum_horizontal(is_zero_loans_over_60)
        .alias("is_zero_loans_over_60")
        .cast(pl.Int8),
        pl.sum_horizontal(is_zero_loans_total)
        .alias("is_zero_loans_total")
        .cast(pl.Int8),
        pl.sum_horizontal(enc_paym_sum_03).alias("enc_paym_sum_03").cast(pl.Int16),
        pl.sum_horizontal(enc_paym_sum_424).alias("enc_paym_sum_424").cast(pl.Int16),
    )
    return df


@print_status_and_time
def grouping_features(input_df: pl.DataFrame) -> pl.DataFrame:
    """
    Apply various grouping transformations to the input DataFrame.

    This function clones the input DataFrame and applies a series of predefined grouping
    functions to it. Each grouping function processes specific features in the DataFrame
    and returns a transformed version of the input.

    Args:
        input_df (pl.DataFrame): Input Polars DataFrame to be transformed.

    Returns:
        pl.DataFrame: A Polars DataFrame with features grouped based on various logic.
    """

    def credit_type_grpd(input_df):
        df = input_df.clone()
        df = df.with_columns(
            pl.when(pl.col("enc_loans_credit_type").is_in([0, 2]))
            .then(0)
            .when(pl.col("enc_loans_credit_type").is_in([1, 7]))
            .then(1)
            .when(pl.col("enc_loans_credit_type").is_in([3, 4]))
            .then(2)
            .otherwise(3)
            .cast(pl.Int8)
            .alias("credit_type_grpd")
        )

        return df.drop("enc_loans_credit_type")

    def cr_cost_rate_grpd(input_df):
        df = input_df.clone()
        df = df.with_columns(
            pl.when(pl.col("pre_loans_credit_cost_rate").is_in([6, 7, 10]))
            .then(2)
            .when(pl.col("pre_loans_credit_cost_rate").is_in([0, 1, 2, 3, 8, 9]))
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("cr_cost_rate_grpd")
        )

        return df.drop("pre_loans_credit_cost_rate")

    def pterm_grpd(input_df):
        df = input_df.clone()
        df = df.with_columns(
            pl.when(pl.col("pre_pterm").is_in([6, 16, 17]))
            .then(0)
            .when(pl.col("pre_pterm").is_in([0, 1, 2, 4, 7, 8, 9, 11, 12, 13, 15]))
            .then(1)
            .otherwise(2)
            .cast(pl.Int8)
            .alias("pterm_grpd")
        )

        return df.drop("pre_pterm")

    def fterm_grpd(input_df):
        df = input_df.clone()
        df = df.with_columns(
            pl.when(pl.col("pre_fterm").is_in([0, 2, 6, 9, 10, 11]))
            .then(0)
            .when(pl.col("pre_fterm").is_in([1, 5, 8, 12, 14, 15]))
            .then(1)
            .when(pl.col("pre_fterm").is_in([4]))
            .then(3)
            .otherwise(2)
            .cast(pl.Int8)
            .alias("fterm_grpd")
        )

        return df.drop("pre_fterm")

    def cr_limit_grpd(input_df):
        df = input_df.clone()
        df = df.with_columns(
            pl.when(pl.col("pre_loans_credit_limit").is_in([0, 2, 11, 15]))
            .then(0)
            .when(pl.col("pre_loans_credit_limit").is_in([7, 8, 9, 14, 16, 17]))
            .then(1)
            .when(pl.col("pre_loans_credit_limit").is_in([1, 3, 10, 12, 19]))
            .then(2)
            .otherwise(3)
            .cast(pl.Int8)
            .alias("cr_limit_grpd")
        )

        return df.drop("pre_loans_credit_limit")

    def pre_util_grpd(input_df):
        df = input_df.clone()
        df = df.with_columns(
            pl.when(pl.col("pre_util").is_in([4, 5, 8, 12, 13]))
            .then(0)
            .when(pl.col("pre_util").is_in([0, 2, 10, 16, 19]))
            .then(1)
            .when(pl.col("pre_util").is_in([1, 7, 11, 15, 18]))
            .then(2)
            .otherwise(3)
            .cast(pl.Int8)
            .alias("pre_util_grpd")
        )

        return df.drop("pre_util")

    def pre_over2limit_grpd(input_df):
        df = input_df.clone()
        df = df.with_columns(
            pl.when(pl.col("pre_over2limit").is_in([2, 5]))
            .then(0)
            .when(pl.col("pre_over2limit").is_in([0]))
            .then(1)
            .when(pl.col("pre_over2limit").is_in([3, 4, 6, 10, 14, 15, 16, 17]))
            .then(2)
            .otherwise(3)
            .cast(pl.Int8)
            .alias("pre_over2limit_grpd")
        )

        return df.drop("pre_over2limit")

    def pre_maxover2limit_grpd(input_df):
        df = input_df.clone()
        df = df.with_columns(
            pl.when(pl.col("pre_maxover2limit").is_in([17]))
            .then(0)
            .when(pl.col("pre_maxover2limit").is_in([0, 3, 4]))
            .then(1)
            .when(
                pl.col("pre_maxover2limit").is_in(
                    [1, 2, 5, 7, 8, 9, 10, 11, 12, 13, 15, 16]
                )
            )
            .then(2)
            .otherwise(3)
            .cast(pl.Int8)
            .alias("pre_maxover2limit_grpd")
        )

        return df.drop("pre_maxover2limit")

    def holder_type_grpd(input_df):
        df = input_df.clone()
        df = df.with_columns(
            pl.when(pl.col("enc_loans_account_holder_type").is_in([2, 3]))
            .then(0)
            .when(pl.col("enc_loans_account_holder_type").is_in([1, 5, 6]))
            .then(1)
            .otherwise(2)
            .cast(pl.Int8)
            .alias("holder_type_grpd")
        )

        return df.drop("enc_loans_account_holder_type")

    def cr_status_grpd(input_df):
        df = input_df.clone()
        df = df.with_columns(
            pl.when(pl.col("enc_loans_credit_status").is_in([1]))
            .then(0)
            .when(pl.col("enc_loans_credit_status").is_in([2, 3]))
            .then(1)
            .otherwise(2)
            .cast(pl.Int8)
            .alias("cr_status_grpd")
        )

        return df.drop("enc_loans_credit_status")

    def acc_cur_grpd(input_df):
        df = input_df.clone()
        df = df.with_columns(
            pl.when(pl.col("enc_loans_account_cur").is_in([0, 2]))
            .then(0)
            .otherwise(1)
            .cast(pl.Int8)
            .alias("acc_cur_grpd")
        )

        return df.drop("enc_loans_account_cur")

    def since_op_grpd(input_df):
        df = input_df.clone()
        df = df.with_columns(
            pl.when(pl.col("pre_since_opened").is_in([1, 2, 3, 11, 17]))
            .then(0)
            .when(pl.col("pre_since_opened").is_in([4, 5, 7, 9, 18]))
            .then(1)
            .when(pl.col("pre_since_opened").is_in([0, 6, 10, 13, 19]))
            .then(2)
            .otherwise(3)
            .cast(pl.Int8)
            .alias("since_op_grpd")
        )

        return df.drop("pre_since_opened")

    def since_confrm_grpd(input_df):
        df = input_df.clone()
        df = df.with_columns(
            pl.when(pl.col("pre_since_confirmed").is_in([0, 1, 5, 12, 13]))
            .then(0)
            .when(pl.col("pre_since_confirmed").is_in([2, 3, 9, 10, 11, 16, 17]))
            .then(1)
            .otherwise(2)
            .cast(pl.Int8)
            .alias("since_confrm_grpd")
        )

        return df.drop("pre_since_confirmed")

    def till_pcl_grpd(input_df):
        df = input_df.clone()
        df = df.with_columns(
            pl.when(pl.col("pre_till_pclose").is_in([0, 2, 8, 12]))
            .then(0)
            .when(pl.col("pre_till_pclose").is_in([1, 11, 14, 15]))
            .then(1)
            .when(pl.col("pre_till_pclose").is_in([5, 7, 10, 16]))
            .then(2)
            .otherwise(3)
            .cast(pl.Int8)
            .alias("till_pcl_grpd")
        )

        return df.drop("pre_till_pclose")

    def till_fcl_grpd(input_df):
        df = input_df.clone()
        df = df.with_columns(
            pl.when(pl.col("pre_till_fclose").is_in([0, 6, 14, 15]))
            .then(0)
            .when(pl.col("pre_till_fclose").is_in([7, 8, 11, 12]))
            .then(1)
            .when(pl.col("pre_till_fclose").is_in([2, 4, 5, 10]))
            .then(2)
            .otherwise(3)
            .cast(pl.Int8)
            .alias("till_fcl_grpd")
        )

        return df.drop("pre_till_fclose")

    def next_pay_summ_grpd(input_df):
        df = input_df.clone()
        df = df.with_columns(
            pl.when(pl.col("pre_loans_next_pay_summ").is_in([5]))
            .then(0)
            .when(pl.col("pre_loans_next_pay_summ").is_in([0, 2, 3]))
            .then(1)
            .otherwise(2)
            .cast(pl.Int8)
            .alias("next_pay_summ_grpd")
        )

        return df.drop("pre_loans_next_pay_summ")

    def outstnd_grpd(input_df):
        df = input_df.clone()
        df = df.with_columns(
            pl.when(pl.col("pre_loans_outstanding").is_in([1, 3]))
            .then(0)
            .otherwise(1)
            .cast(pl.Int8)
            .alias("outstnd_grpd")
        )

        return df.drop("pre_loans_outstanding")

    def tot_overdue_grpd(input_df):
        df = input_df.clone()
        df = df.with_columns(
            pl.when(pl.col("pre_loans_total").is_in([23, 25, 26, 29, 39]))
            .then(0)
            .when(pl.col("pre_loans_total").is_in([30, 33, 35, 36, 38, 41, 47, 52, 53]))
            .then(1)
            .otherwise(2)
            .cast(pl.Int8)
            .alias("tot_overdue_grpd")
        )

        return df.drop("pre_loans_total")

    def max_overdue_sum_grpd(input_df):
        df = input_df.clone()
        df = df.with_columns(
            pl.when(pl.col("pre_loans_max_overdue_sum").is_in([2]))
            .then(0)
            .when(pl.col("pre_loans_max_overdue_sum").is_in([3]))
            .then(1)
            .otherwise(2)
            .cast(pl.Int8)
            .alias("max_overdue_sum_grpd")
        )

        return df.drop("pre_loans_max_overdue_sum")

    df = input_df.clone()

    df = credit_type_grpd(df)
    df = cr_cost_rate_grpd(df)
    df = pterm_grpd(df)
    df = fterm_grpd(df)
    df = cr_limit_grpd(df)
    df = pre_util_grpd(df)
    df = pre_over2limit_grpd(df)
    df = pre_maxover2limit_grpd(df)
    df = holder_type_grpd(df)
    df = cr_status_grpd(df)
    df = acc_cur_grpd(df)
    df = since_op_grpd(df)
    df = since_confrm_grpd(df)
    df = till_pcl_grpd(df)
    df = till_fcl_grpd(df)
    df = next_pay_summ_grpd(df)
    df = outstnd_grpd(df)
    df = tot_overdue_grpd(df)
    df = max_overdue_sum_grpd(df)

    return df


@print_status_and_time
def combined_features(input_df: pl.DataFrame) -> pl.DataFrame:
    """
    Combine specific features in the DataFrame and add new derived features.

    This function clones the input DataFrame and computes several new features by
    combining specific columns through addition. The results are stored in new columns
    and cast to `Int8` data type. The function also drops certain columns from the DataFrame
    that are no longer needed.

    Args:
        input_df (pl.DataFrame): Input Polars DataFrame containing the features to be combined.

    Returns:
        pl.DataFrame: A Polars DataFrame with the newly created combined features and
                      unnecessary columns removed.

    Note:
        - New features are created as follows:
            - `util_no_overdue`: Sum of `pre_util_grpd` and `is_zero_loans_total`.
            - `maxoverdue_till_pclose`: Sum of `pre_maxover2limit_grpd` and `till_pcl_grpd`.
            - `delays_till_end`: Sum of `till_fcl_grpd` and `tot_overdue_grpd`.
            - `over60_totoverdue`: Sum of `is_zero_loans_over_60` and `tot_overdue_grpd`.
            - `upto60_fcl`: Sum of `is_zero_loans_upto_60` and `till_fcl_grpd`.
            - `outstnd_grpd + is0_tot`: Sum of `outstnd_grpd` and `is_zero_loans_total`.
        - Ensure that the columns used in the transformations (`pre_util_grpd`,
          `is_zero_loans_total`, etc.) exist in the input DataFrame.
        - The columns to be dropped are: `pre_loans_cols`, `is_zero_loans_total`,
          `enc_paym_sum_03`, `enc_paym_sum_424`.
    """

    df = input_df.clone()
    df = df.with_columns(
        ((pl.col("pre_util_grpd")) + (pl.col("is_zero_loans_total")))
        .alias("util_no_overdue")
        .cast(pl.Int8),
        ((pl.col("pre_maxover2limit_grpd")) + (pl.col("till_pcl_grpd")))
        .alias("maxoverdue_till_pclose")
        .cast(pl.Int8),
        ((pl.col("till_fcl_grpd")) + (pl.col("tot_overdue_grpd")))
        .alias("delays_till_end")
        .cast(pl.Int8),
        ((pl.col("is_zero_loans_over_60")) + (pl.col("tot_overdue_grpd")))
        .alias("over60_totoverdue")
        .cast(pl.Int8),
        ((pl.col("is_zero_loans_upto_60")) + (pl.col("till_fcl_grpd")))
        .alias("upto60_fcl")
        .cast(pl.Int8),
        ((pl.col("outstnd_grpd")) + (pl.col("is_zero_loans_total")))
        .alias("outstnd_grpd + is0_tot")
        .cast(pl.Int8),
    )

    cols_to_drop = (
        pre_loans_cols + is_zero_loans_total + enc_paym_sum_03 + enc_paym_sum_424
    )

    return df.drop(cols_to_drop)


@print_status_and_time
def pl_featuretools(input_df: pl.DataFrame) -> pl.DataFrame:
    """
    Generate aggregated features for boolean and numeric columns in the DataFrame.

    This function processes the input DataFrame and calculates a variety of aggregations
    for both boolean and numeric columns. The boolean columns are aggregated by calculating
    the percentage of `True` values and the count of occurrences. For numeric columns, several
    statistics are computed, such as minimum, mean, median, variance, maximum, sum, and standard
    deviation. The resulting features are grouped by the `id` column and returned in a new DataFrame.

    Args:
        input_df (pl.DataFrame): Input Polars DataFrame containing the columns to be aggregated.

    Returns:
        pl.DataFrame: A Polars DataFrame with aggregated features for each `id`.

    Note:
        - The function handles boolean and numeric columns differently. Boolean columns are
          aggregated by calculating the percentage of `True` values and the count. Numeric columns
          are aggregated by calculating various statistics such as mean, sum, min, max, etc.
        - The `id` column is used for grouping the DataFrame, and the resulting DataFrame is sorted
          by the `id`.
        - The columns `id` and the excluded numeric columns are not included in the aggregation.
    """

    df = input_df.clone()

    boolean_cols = [
        col for col, dtype in zip(df.columns, df.dtypes) if dtype == pl.Boolean
    ]
    numeric_cols = [
        col
        for col, dtype in zip(df.columns, df.dtypes)
        if dtype in [pl.Float32, pl.Float64, pl.Int64, pl.Int32, pl.Int16, pl.Int8]
        and col not in ["id"]
    ]

    boolean_aggregations = []

    for col in boolean_cols:
        boolean_aggregations.extend(
            [
                pl.col(col).mean().alias(f"{col}_percent_true").cast(pl.Float32),
                pl.col(col).count().alias(f"{col}_count").cast(pl.Int8),
            ]
        )

    numeric_aggregations = []
    for col in numeric_cols:
        numeric_aggregations.extend(
            [
                pl.col(col).min().alias(f"{col}_min").cast(pl.Int8),
                pl.col(col).mean().alias(f"{col}_mean").cast(pl.Float32),
                pl.col(col).median().alias(f"{col}_median").cast(pl.Float32),
                pl.col(col).var().alias(f"{col}_var").cast(pl.Float32),
                pl.col(col).max().alias(f"{col}_max").cast(pl.Int8),
                pl.col(col).sum().alias(f"{col}_sum").cast(pl.Int16),
                pl.col(col).std().alias(f"{col}_std").cast(pl.Float32),
            ]
        )

    compressed_df = df.group_by("id").agg(boolean_aggregations + numeric_aggregations)

    return compressed_df.sort("id")
