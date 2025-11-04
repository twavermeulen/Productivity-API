
# python
from fastapi import FastAPI
from typing import List, Optional
from pydantic import BaseModel
from xgboost import XGBRegressor
import numpy as np
import shap

class PredictRequest(BaseModel):
    job_satisfaction_score: Optional[float] = 0.0
    days_feeling_burnout_per_month: Optional[float] = 0.0
    daily_social_media_time: Optional[float] = 0.0
    age: Optional[float] = 0.0
    sleep_hours: Optional[float] = 0.0
    breaks_during_work: Optional[float] = 0.0
    gender: Optional[str] = 'Other'

model = XGBRegressor()
model.load_model('model.json')

explainer = shap.Explainer(model)

app = FastAPI()

_gender_aliases = {
    'male': 'Male',
    'm': 'Male',
    'female': 'Female',
    'f': 'Female',
    'other': 'Other',
    'o': 'Other'
}

@app.post("/predict/")
def predict(provided: PredictRequest):
    data = provided.dict()
    raw_gender = data.get('gender') or 'Other'
    gender_key = _gender_aliases.get(raw_gender.lower(), raw_gender.capitalize())

    # assume feature names are present and in the trained order
    booster = model.get_booster()
    model_feature_names = booster.feature_names

    x_row = []
    for fname in model_feature_names:
        if fname == 'gender':
            ord_map = {'Male': 0.0, 'Female': 1.0, 'Other': 2.0}
            x_row.append(float(ord_map[gender_key]))
        elif fname.startswith('gender_'):
            suffix = fname.split('gender_', 1)[1]
            x_row.append(1.0 if suffix.lower() == gender_key.lower() else 0.0)
        else:
            x_row.append(float(data.get(fname)))

    x = np.array([x_row], dtype=float)

    prediction = float(model.predict(x)[0])

    shap_exp = explainer(x)
    contribs = shap_exp.values[0]  # assume shape (1, n_features)

    contributions = {name: round(float(val), 3) for name, val in zip(model_feature_names, contribs)}

    top_idx = np.argsort(np.abs(contribs))[::-1][:3]
    top_contributors = [
        {"feature": model_feature_names[i], "contribution": round(float(contribs[i]), 3)}
        for i in top_idx
    ]

    return {
        "productivity_score": round(prediction, 1),
        "top_contributors": top_contributors,
        "all_contributions": contributions
    }
