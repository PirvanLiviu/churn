# churn

Predicts whether a telecom customer will churn, using an XGBoost model trained on the Telco customer churn dataset.

## Run with Docker

```
docker compose up --build
```

- Web UI (Gradio): http://localhost:8000/ui
- API docs (Swagger): http://localhost:8000/docs
- Health check: http://localhost:8000/health

Example request:

```
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":1,"PhoneService":"No","MultipleLines":"No phone service","InternetService":"DSL","OnlineSecurity":"No","OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":29.85,"TotalCharges":29.85}'
```

`POST /predict/batch` takes a list of the same objects. The churn threshold defaults to 0.5 and can be changed with the `THRESHOLD` environment variable in `docker-compose.yml`.

## Run locally

```
pip install -r requirements.txt
uvicorn api.main:app --reload
```

## Preprocess and train

```
python src/preprocessor.py              # data/raw.csv -> data/preprocessed.csv
python src/model.py                     # tune (50 trials) + train -> models/model_v1.ubj
python src/model.py --trials 100 --version 2
python src/evaluate.py                  # scores on the held-out test set (run from src/)
```

Tuning uses 5-fold cross-validation on the training split only, so the test set stays unseen until `evaluate.py`. The chosen hyperparameters are saved next to the model as `models/model_v<version>_params.json`.

## Tests

```
python -m unittest discover -s tests -v
```
