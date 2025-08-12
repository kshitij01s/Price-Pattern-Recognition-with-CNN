import os
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

load_model("C:/Users/admin/Desktop/price_pattern_cnn/models/pattern_detector.h5")


# === Config Paths ===
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "pattern_detector.h5"
DATA_DIR = BASE_DIR / "data" / "candlestick_images"

# === Load Class Labels from Folder Names ===
classes = sorted([folder for folder in os.listdir(DATA_DIR) if os.path.isdir(DATA_DIR / folder)])
print("Looking for model at:", MODEL_PATH)
print("Does it exist?", MODEL_PATH.exists())
print("Size (bytes):", MODEL_PATH.stat().st_size if MODEL_PATH.exists() else "File not found")

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))  # Resize to 128x128
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize if your model was trained with normalized inputs
    return img_array

def predict(img_path):
    # Check if model file exists
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

    # Load the trained model
    model = load_model(MODEL_PATH)

    # Load and preprocess the image
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
    except Exception as e:
        raise ValueError(f"Error loading image: {e}")

    # Predict the pattern
    preds = model.predict(img_array)
    conf = float(np.max(preds))
    label = classes[int(np.argmax(preds))]

    return label, conf

# === Run a Test ===
if __name__ == "__main__":
    test_image = DATA_DIR / "Head & Shoulders" / "sample.png"
    if not test_image.exists():
        print(f"Test image not found at: {test_image}")
    else:
        pattern, confidence = predict(test_image)
        print(f"Pattern: {pattern}, Confidence: {confidence:.2f}")
