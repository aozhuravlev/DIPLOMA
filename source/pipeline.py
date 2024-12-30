import time

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

from data_loading_functions import load_data, load_target
from cleaning_functions import remove_hi_corr_feats, convert_to_pandas
from modeling_functions import get_model_and_score, dump_model
from feature_engineering_pl import (
    horizontal_sum,
    grouping_features,
    combined_features,
    pl_featuretools,
)


def main():
    """
    Main function to execute the end-to-end data processing and modeling pipeline.

    This function performs the following steps:
    1. Load the raw data using `load_data`.
    2. Apply feature engineering transformations:
        - Calculate horizontal sums of features using `horizontal_sum`.
        - Group features based on defined logic using `grouping_features`.
        - Combine features with specific transformations using `combined_features`.
        - Generate additional features using feature tools with `pl_featuretools`.
    3. Clean the data and load the target variable:
        - Remove highly correlated features using `remove_hi_corr_feats`.
        - Load the target variable using `load_target`.
        - Convert the data to a pandas DataFrame using `convert_to_pandas`.
    4. Apply machine learning modeling:
        - Train the model and compute its score using `get_model_and_score`.
    5. Execute the pipeline and save the trained model, along with its score and execution time.

    Dependencies:
    - Libraries: `time`, `dlf`, `fepl`, `clf`, `mlf`, and `sklearn.pipeline.Pipeline`.

    Outputs:
    - The trained model and score are saved using `dump_model`.

    Execution:
    - The function is executed when the script is run directly.
    """

    start_time = time.time()

    df_data = load_data()

    feature_engineering = Pipeline(
        steps=[
            ("horizontal_sum", FunctionTransformer(horizontal_sum)),
            ("grouping_features", FunctionTransformer(grouping_features)),
            ("combined_features", FunctionTransformer(combined_features)),
            ("pl_featuretools", FunctionTransformer(pl_featuretools)),
        ]
    )

    cleaning_and_target_loading = Pipeline(
        steps=[
            ("remove_hi_corr_feats", FunctionTransformer(remove_hi_corr_feats)),
            ("target_loading", FunctionTransformer(load_target)),
            ("convert_to_pandas", FunctionTransformer(convert_to_pandas)),
        ]
    )

    modeling = Pipeline(
        steps=[
            ("ml_model", FunctionTransformer(get_model_and_score)),
        ]
    )

    pipe = Pipeline(
        steps=[
            ("feature_engineering", feature_engineering),
            ("cleaning_and_target_loading", cleaning_and_target_loading),
            ("modeling", modeling),
        ]
    )

    model, score = pipe.fit_transform(df_data)

    dump_model(model, score, start_time)


if __name__ == "__main__":
    main()
