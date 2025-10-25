# Productivity Score API

## Dependencies
- `conda install -c conda-forge fastapi uvicorn xgboost numpy -y`

## Run the API
- ` uvicorn model:app --reload`

## Test the API
POST http://127.0.0.1:8000/predict


### Sample input
```json
{
    "job_satisfaction_score": 6.0,
    "days_feeling_burnout_per_month": 2.0,
    "daily_social_media_time": 60.0,
    "age": 40.0,
    "sleep_hours": 7.0,
    "breaks_during_work": 2.0
}
```
