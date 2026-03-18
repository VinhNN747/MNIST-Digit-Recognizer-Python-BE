import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset
from torchvision import datasets, transforms

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PixelInput(BaseModel):
    pixels: list


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # 28x28 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 14x14

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # -> 7x7
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN()
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
model.to(device)
model.eval()


@app.post("/predict")
def predict_digit(data: PixelInput):
    # 1. convert input
    x = np.array(data.pixels).astype('float32') / 255.0

    # 2. reshape đúng format CNN
    x = torch.tensor(x).view(1, 1, 28, 28).to(device)

    # 3. predict
    with torch.no_grad():
        output = model(x)

        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, dim=1)

    # 4. trả kết quả
    return {
        "digit": pred.item(),
        "confidence": float(conf.item())
    }
