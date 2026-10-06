from typing import get_args

import gradio as gr

from api.predictor import predict
from api.schemas import Customer

FIELDS = Customer.model_fields
EXAMPLE = Customer.model_config["json_schema_extra"]["example"]
NUMBER_FIELDS = {"tenure", "MonthlyCharges", "TotalCharges"}


def _input(name):
    if name in NUMBER_FIELDS:
        return gr.Number(label=name, value=EXAMPLE[name], minimum=0, precision=0 if name == "tenure" else 2)

    choices = list(get_args(FIELDS[name].annotation))
    return gr.Dropdown(label=name, choices=choices, value=EXAMPLE[name])


def _predict(*values):
    # validate through the same schema the api uses
    customer = Customer(**dict(zip(FIELDS, values)))
    result = predict([customer.model_dump()])[0]
    p = result["churn_probability"]

    return {"Churn": p, "No churn": 1 - p}


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Churn predictor") as ui:
        gr.Markdown("# Customer churn predictor\nFill in the customer details and press **Predict**.")

        names = list(FIELDS)
        inputs = []
        with gr.Row():
            # spread the 19 inputs over 3 columns
            for i in range(3):
                with gr.Column():
                    inputs += [_input(n) for n in names[i::3]]
        # keep inputs in schema order so they line up with FIELDS in _predict
        inputs.sort(key=lambda c: names.index(c.label))

        button = gr.Button("Predict", variant="primary")
        output = gr.Label(label="Prediction")
        button.click(_predict, inputs=inputs, outputs=output)

    return ui
