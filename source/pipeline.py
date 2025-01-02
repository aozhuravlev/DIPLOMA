import time

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

from functions.data_loading_functions import load_data, load_target
from functions.cleaning_functions import (
    drop_constant_features, 
    remove_hi_corr_feats, 
    convert_to_pandas
)
from functions.modeling_functions import dump_model, get_model_and_score
from functions.feature_engineering_pl import (
    horizontal_sum,
    grouping_features,
    combined_features,
    pl_featuretools,
)


def create_preprocessor() -> Pipeline:
    """
    Creates a preprocessing pipeline for feature engineering and data cleaning.

    The pipeline consists of:
    1. Feature engineering steps:
        - Dropping constant features.
        - Computing horizontal sums.
        - Grouping features.
        - Combining features.
        - Applying feature engineering tools with Polars.
    2. Cleaning steps:
        - Removing highly correlated features.
        - Converting the data to a pandas DataFrame.

    Returns:
        Pipeline: A scikit-learn pipeline that combines feature engineering 
                  and cleaning steps for preprocessing.
    """
    feature_engineering = Pipeline(
        steps=[
            ("drop_constant_features", FunctionTransformer(drop_constant_features)),
            ("horizontal_sum", FunctionTransformer(horizontal_sum)),
            ("grouping_features", FunctionTransformer(grouping_features)),
            ("combined_features", FunctionTransformer(combined_features)),
            ("pl_featuretools", FunctionTransformer(pl_featuretools)),
        ]
    )

    cleaning = Pipeline(
        steps=[
            ("remove_hi_corr_feats", FunctionTransformer(remove_hi_corr_feats)),
            ("convert_to_pandas", FunctionTransformer(convert_to_pandas)),
        ]
    )

    preprocessor = Pipeline(
        steps=[
            ("feature_engineering", feature_engineering),
            ("cleaning", cleaning),
        ]
    )

    return preprocessor


def main() -> None:
    """
    Main function to preprocess data, train the model, and save the results.

    Steps:
    1. Record the start time.
    2. Create a preprocessing pipeline using `create_preprocessor`.
    3. Load and preprocess the data.
    4. Train the model and calculate the score.
    5. Save the trained model along with metadata.

    Returns:
        None
    """
    start_time = time.time()
    preprocessor = create_preprocessor()
    
    X = preprocessor.fit_transform(load_data())
    y = convert_to_pandas(load_target())
    
    model, score = get_model_and_score(X, y)
    dump_model(model, score, start_time)

if __name__ == "__main__":
    main()
