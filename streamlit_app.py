import sys
import os

# --- Ensure project root is in PYTHONPATH ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Now safe to import from src ---
from src.predict import predict
import streamlit as st

st.title("Price Pattern Recognition with CNN")

uploaded_file = st.file_uploader("Upload a candlestick chart image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Save temporarily
    temp_path = os.path.join(project_root, "temp_uploaded_image.png")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Run prediction
    pattern, confidence = predict(temp_path)

    st.subheader("Prediction")
    st.write(f"Pattern: **{pattern}**")
    st.write(f"Confidence: **{confidence:.2f}**")

