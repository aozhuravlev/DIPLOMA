# Credit Risk Management Model

## Description

This project focuses on developing a credit risk management model to predict the likelihood of client default on a loan. Financial institutions rely on such models to make informed decisions and mitigate risks associated with loan defaults. By automating the evaluation process, banks can efficiently assess credit applications based on client data, such as employment details, repayment history, and credit type. This reduces the time needed for decision-making and allows experts to audit the model’s output when necessary. The model in this project achieves a ROC-AUC score of 0.76, surpassing the required threshold of 0.75, ensuring robust and accurate predictions. This implementation leverages state-of-the-art machine learning techniques to assist in minimizing financial losses while enhancing operational efficiency.

---

## Project Structure

- **`data/`**
    - **`train_data/`**: Source parquet files containing training data for model development.
    - **`test_files/`**: Parquet files used for model evaluation and testing.
    - **`feats_to_drop.txt`**: List of features to exclude from the model, obtained through XGBoost analysis.
    - **`best_params_3M_pl.csv`**: Optimized parameters for the model, obtained using the Optuna library with 50 trials.
    - **`optuna_report.csv`**: Results from hyperparameter optimization.
- **`media/`**
    - **`roc_curve.png`**: ROC curve visualization.
    - **`confusion_matrix.png`**: Confusion matrix.
    - **`finished_.mp3`**: Audio file played on pipeline completion.
- **`models/`**
    - **`credit_risk_management_model.pkl`**: Serialized model used for predictions.
- **`notebooks/`**
    - Jupyter notebooks used during model development and experimentation.
- **`src/`**
    - **`main.py`**: The main FastAPI application file for serving the model.
    - **`pipeline.py`**: The data processing pipeline script.
    - **`functions/`**: Utility scripts for data cleaning, feature engineering, and model training.
- **`environment.yml`**: Dependencies required to run the application.
- **`requirements.txt`**: Python libraries required for the project.
- **`README.md`**: This guide for using the service.

---

## Endpoints

- **`/status`**: Returns the operational status of the service.
- **`/version`**: Provides the version and metadata of the deployed model.
- **`/predict`**: Predicts client default based on provided data.

---

## Why ROC-AUC?

We chose ROC-AUC as the primary evaluation metric because of the significant class imbalance (1:30) in the dataset. ROC-AUC considers both the true positive rate and false positive rate, making it ideal for binary classification tasks with imbalance. This ensures the model is sensitive to positive cases while avoiding excessive false positives. Our model achieved a ROC-AUC score of **0.76**, exceeding the required threshold of **0.75**.

---

## Feature Importance

Key features used in the model include:

- **`enc_paym_{0...24}`**: Sum of these features provides a significant predictive signal.
- **`is_zero_loans{0...90}`**: Sum of these features is another crucial indicator.
- **`enc_loans_credit_type`**: Encoded type of the loan.

---

## Running the Application

### Installing Dependencies

To run the application, create a virtual environment using Conda and install all required libraries:

```bash
conda env create -f environment.yml
conda activate <your_environment_name>
```

Alternatively, install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Starting the Service

Launch the FastAPI service:

```bash
uvicorn source.main:app --reload
```

The service will be available at `http://127.0.0.1:8000`.

---

## Usage

Input data: The model expects a Parquet file with credit application information for a client.

Example response:

```json
{
    "id": 967015,
    "predicted_default": 0
}
```

- **`0`**: Client is not likely to default.
- **`1`**: Client is likely to default.

---

## Model & Model Evaluation

We used **CatBoostClassifier**, which outperformed other algorithms like XGBoost, LightGBM, and ensemble stacks.

- **ROC-AUC Score**: 0.76
- **Confusion Matrix**:
    - True Positives (TP): 572,795
    - True Negatives (TN): 1,527
    - False Positives (FP): 5,917
    - False Negatives (FN): 19,761

---

## Support & Feedback

If you have any questions or encounter issues, feel free to contact the developer, **Aleksey Zhuravlev**, at **[a.o.zhuravlev@gmail.com](mailto:a.o.zhuravlev@gmail.com)**.