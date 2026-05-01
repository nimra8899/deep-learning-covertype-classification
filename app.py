import streamlit as st
import torch
from model import MLP
import numpy as np

st.title("🌲 Forest Cover Type Predictor")

# Load Model
model = MLP()
model.load_state_dict(torch.load("best_model_seed_0.pt", map_location=torch.device("cpu")))
model.eval()

# Input features
inputs = []

for i in range(54):
    val = st.number_input(f"Feature {i+1}", value=0.0)
    inputs.append(val)

if st.button("Predict"):
    x = torch.tensor([inputs], dtype=torch.float32)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)

    st.write("Class Probabilities:")
    st.write(probs.numpy())