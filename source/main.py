import dill
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

# from typing import Optional


app = FastAPI()
with open("models/credit_risk_management_model.pkl", "rb") as file:
    model = dill.load(file)


class Form(BaseModel):
    id: int
    rn: int
    pre_since_opened: int
    pre_since_confirmed: int
    pre_pterm: int
    pre_fterm: int
    pre_till_pclose: int
    pre_till_fclose: int
    pre_loans_credit_limit: int
    pre_loans_next_pay_summ: int
    pre_loans_outstanding: int
    pre_loans_max_overdue_sum: int
    pre_loans_credit_cost_rate: int
    pre_loans5: int
    pre_loans530: int
    pre_loans3060: int
    pre_loans6090: int
    pre_loans90: int
    is_zero_loans5: int
    is_zero_loans530: int
    is_zero_loans3060: int
    is_zero_loans6090: int
    is_zero_loans90: int
    pre_util: int
    pre_over2limit: int
    pre_maxover2limit: int
    is_zero_util: int
    is_zero_over2limit: int
    is_zero_maxover2limit: int
    enc_paym_0: int
    enc_paym_1: int
    enc_paym_2: int
    enc_paym_3: int
    enc_paym_4: int
    enc_paym_5: int
    enc_paym_6: int
    enc_paym_7: int
    enc_paym_8: int
    enc_paym_9: int
    enc_paym_10: int
    enc_paym_11: int
    enc_paym_12: int
    enc_paym_13: int
    enc_paym_14: int
    enc_paym_15: int
    enc_paym_16: int
    enc_paym_17: int
    enc_paym_18: int
    enc_paym_19: int
    enc_paym_20: int
    enc_paym_21: int
    enc_paym_22: int
    enc_paym_23: int
    enc_paym_24: int
    enc_loans_account_holder_type: int
    enc_loans_credit_status: int
    enc_loans_credit_type: int
    enc_loans_account_cur: int
    pclose_flag: int
    fclose_flag: int


class Prediction(BaseModel):
    id: int
    predicted_default: int


@app.get("/status")
def status():
    return "I await your commands, my Lord."


@app.get("/version")
def version():
    return model["metadata"]


@app.post("/predict", response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.model_dump()])
    y = model["model"].predict(df)

    return {
        "client_id": form.id,
        "predicted_default": y[0],
    }
