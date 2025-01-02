import time

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

from data_loading_functions import load_data, load_target
from cleaning_functions import (
    drop_constant_features, 
    remove_hi_corr_feats, 
    convert_to_pandas
)
from modeling_functions import dump_model, get_model_and_score
from feature_engineering_pl import (
    horizontal_sum,
    grouping_features,
    combined_features,
    pl_featuretools,
)


def create_preprocessor():
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


def main():
    start_time = time.time()
    preprocessor = create_preprocessor()
    
    X = preprocessor.fit_transform(load_data())
    y = convert_to_pandas(load_target())
    
    model, score = get_model_and_score(X, y)
    dump_model(model, score, start_time)


if __name__ == "__main__":
    main()
