import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# load model
data = np.load("model.npz", allow_pickle=True)
W = data["weights"]
b = data["biases"]


class PixelInput(BaseModel):
    pixels: list


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def softmax(z):
    e = np.exp(z - np.max(z))
    return e / np.sum(e)


def predict(x):
    a = x.reshape(784, 1)

    for Wi, bi in zip(W[:-1], b[:-1]):
        a = sigmoid(Wi @ a + bi)

    z = W[-1] @ a + b[-1]
    a = softmax(z)

    return int(np.argmax(a))


@app.post("/predict")
def predict_digit(data: PixelInput):
    x = np.array(data.pixels)

    digit = predict(x)

    return {"digit": digit}
