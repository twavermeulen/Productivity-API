from fastapi import FastAPI
from typing import Dict, List
from xgboost import XGBRegressor
import xgboost as xgb
import numpy as np
import shap

model = XGBRegressor()
model.load_model('xgb_model.json')

explainer = shap.Explainer(model)

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


    shap_exp = explainer(x)
    shap_vals = shap_exp.values

    if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 2 and shap_vals.shape[0] == 1:
        contribs = shap_vals[0]
    else:
        contribs = np.ravel(shap_vals)

    contributions = {name: round(float(val), 3) for name, val in zip(features, contribs)}

    top_idx = np.argsort(np.abs(contribs))[::-1][:3]
    top_contributors = []
    for i in top_idx:
        val = float(contribs[i])
        top_contributors.append({
            "feature": features[i],
            "contribution": round(val, 3),
        })


    return {
        "productivity_score": round(prediction, 1),
        "top_contributors": top_contributors,
        "all_contributions": contributions
    }
