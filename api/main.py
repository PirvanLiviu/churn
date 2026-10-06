from contextlib import asynccontextmanager

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from api.predictor import MODEL_V, load_model, predict
from api.schemas import Customer, Prediction
from api.ui import build_ui


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load the model once at startup instead of on the first request
    load_model()
    yield


app = FastAPI(title="Churn prediction API", version=f"{MODEL_V}", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/ui")


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_V}


@app.post("/predict", response_model=Prediction)
def predict_one(customer: Customer):
    return predict([customer.model_dump()])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(customers: list[Customer]):
    if not customers:
        return []
    return predict([c.model_dump() for c in customers])


app = gr.mount_gradio_app(app, build_ui(), path="/ui")
