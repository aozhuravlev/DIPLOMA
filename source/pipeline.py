import time

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from data_loading_functions import load_data, load_target
from cleaning_functions import remove_hi_corr_feats, convert_to_pandas
from modeling_functions import get_model_and_score, dump_model, load_model
from feature_engineering_pl import (
    horizontal_sum,
    grouping_features,
    combined_features,
    pl_featuretools,
)


def main():

    start_time = time.time()

    feature_engineering = Pipeline(
        steps=[
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


    X = preprocessor.fit_transform(load_data())
    y = convert_to_pandas(load_target())

    
    model = load_model()

    pipe = Pipeline(steps=[("model", model)])      

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=137, shuffle=True
    )

    fitted_pipe = pipe.fit(X_train, y_train)
    score = roc_auc_score(y_test, fitted_pipe.predict_proba(X_test)[:, 1])
    
    dump_model(fitted_pipe, score, start_time)


if __name__ == "__main__":
    main()
