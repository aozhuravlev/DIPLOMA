import time
import dill
import datetime
import pygame

import pandas as pd

from typing import Tuple
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from feature_engineering_pl import print_status_and_time


@print_status_and_time
def get_model_and_score(input_df: pd.DataFrame) -> Tuple[CatBoostClassifier, float]:
    """
    Train a CatBoost classifier on the input data and evaluate its performance.

    This function splits the input DataFrame into features (`X`) and target (`y`),
    then splits the data into training and test sets. It reads the best hyperparameters
    from a CSV file and trains a CatBoost classifier. After training, the function
    evaluates the model's performance using AUC (Area Under the Curve) on the test set
    and returns the trained model along with the AUC score.

    Args:
        input_df (pd.DataFrame): Input pandas DataFrame containing the features and target.

    Returns:
        Tuple[CatBoostClassifier, float]: The trained CatBoost model and its AUC score.

    Note:
        - The `flag` column is assumed to be the target variable.
        - The function expects the best hyperparameters for the model to be in the
          `best_params_3M_pl.csv` file located at `../data/`.
        - The model is trained using a GPU for faster computation.
        - The AUC score is calculated using the predicted probabilities of the positive class.
    """

    df = input_df.copy()

    X = df.drop("flag", axis=1)
    y = df["flag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=137, shuffle=True
    )

    params_df = pd.read_csv("../data/best_params_3M_pl.csv")
    params = params_df.to_dict(orient="records")[0]

    cbc = CatBoostClassifier(
        **params,
        random_seed=137,
        eval_metric="AUC",
        task_type="GPU",
        logging_level="Silent",
        auto_class_weights="SqrtBalanced",
    )

    cbc.fit(X_train, y_train)
    y_pred = cbc.predict_proba(X_test)[:, 1]

    score = roc_auc_score(y_test, y_pred)

    return cbc, score


def dump_model(model: CatBoostClassifier, score: float, start_time: float) -> None:
    """
    Save the trained model and its metadata to a file, and play a sound when finished.

    This function computes the total training time, saves the model and metadata (including
    model type, accuracy, and processing time) to a `.pkl` file, and plays a sound to signal
    the completion. It prints the model's ROC-AUC score and the elapsed time.

    Args:
        model (Any): The trained model to be saved.
        score (float): The ROC-AUC score of the model.
        start_time (float): The start time of the model training process, used to calculate
                            elapsed time.

    Returns:
        None

    Note:
        - The model is saved in the file `../models/credit_risk_management_model.pkl`.
        - The metadata includes the model's name, author, version, date, type, accuracy,
          and processing time.
        - The function uses `pygame` to play a sound from `../media/finished_.mp3`
          when the process is complete.
    """

    def tot_time(start_time: float) -> str:
        tot_time = time.time() - start_time
        mins = int(tot_time // 60)
        secs = int(tot_time % 60)
        return f"{mins} min {secs} sec"

    def playsound() -> None:
        pygame.init()
        pygame.mixer.music.load("../media/finished_.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

    time_str = tot_time(start_time)

    with open("../models/credit_risk_management_model.pkl", "wb") as file:
        dill.dump(
            {
                "model": model,
                "metadata": {
                    "name": "Credit risk management model",
                    "author": "Aleksey Zhuravlev",
                    "version": 1,
                    "date": datetime.datetime.now(),
                    "type": type(model).__name__,
                    "accuracy": f"{score:.2%}",
                    "processing time": time_str,
                },
            },
            file,
            recurse=True,
        )

    print(
        f"\nModel has been created.\nROC-AUC score: {score:.2%}\nTime elapsed: {time_str}"
    )

    playsound()
