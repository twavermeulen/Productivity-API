from fastapi import FastAPI
from typing import Dict, List
from xgboost import XGBRegressor
import xgboost as xgb
import numpy as np

model = XGBRegressor()
model.load_model('xgb_model.json')

app = FastAPI()

features: List[str] = [
    'job_satisfaction_score',
    'days_feeling_burnout_per_month',
    'daily_social_media_time',
    'age',
    'sleep_hours',
    'breaks_during_work',
]


@app.post("/predict/")
def predict(provided: Dict[str, float]):
    x = np.array([[float(provided[n]) for n in features]])

    prediction = float(model.predict(x)[0])

    contribs = model.get_booster().predict(xgb.DMatrix(x, feature_names=features), pred_contribs=True)[0][:-1]
    top_contributors = [name for name, _ in sorted(zip(features, contribs), key=lambda t: abs(t[1]), reverse=True)[:3]]

    return {
        "productivity_score": round(prediction, 1),
        "top_contributors": top_contributors
    }