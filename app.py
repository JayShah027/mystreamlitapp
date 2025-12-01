# app.py (REPLACE your existing file with this)
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import os
import zipfile
import gdown
from pathlib import Path

st.set_page_config(page_title="Image Classifier (Streamlit Cloud)", layout="wide")

# Paths we expect after extraction or if committed to repo
MODELS_ROOT = Path("models")
LABELS_FILENAME = "labels.json"

def debug_print(msg):
    # Also print to logs
    st.write(msg)
    print(msg)

def find_model_path():
    """
    Search the models folder for a usable model:
      - a folder containing saved_model.pb (TensorFlow SavedModel)
      - a .keras or .h5 file (Keras single-file format)
      - folder named 'final_saved_model' (common)
    Returns (model_path: Path or None, labels_path: Path or None)
    """
    if not MODELS_ROOT.exists():
        return None, None

    # Check labels
    labels_candidates = list(MODELS_ROOT.rglob(LABELS_FILENAME))
    labels_path = labels_candidates[0] if labels_candidates else None

    # 1) Check for SavedModel folders (has saved_model.pb)
    for d in MODELS_ROOT.rglob("*"):
        if d.is_dir() and (d / "saved_model.pb").exists():
            return d, labels_path

    # 2) Check for .keras or .h5 files
    for f in MODELS_ROOT.rglob("*.keras"):
        return f, labels_path
    for f in MODELS_ROOT.rglob("*.h5"):
        return f, labels_path

    # 3) Common folder name
    candidate = MODELS_ROOT / "final_saved_model"
    if candidate.exists():
        return candidate, labels_path

    # Not found
    return None, labels_path

def download_and_extract_from_drive(file_id):
    zip_path = MODELS_ROOT / "model.zip"
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    debug_print(f"Downloading model from Drive id: {file_id} ...")
    gdown.download(url, str(zip_path), quiet=False)
    debug_print(f"Downloaded to {zip_path}. Now extracting...")
    with zipfile.ZipFile(str(zip_path), "r") as z:
        z.extractall(str(MODELS_ROOT))
    try:
        zip_path.unlink()
    except Exception:
        pass
    debug_print("Extraction complete. Files now in /models/")

@st.cache_resource
def ensure_model_available():
    # 0) If already in repo or previously extracted, detect it
    model_path, labels_path = find_model_path()
    if model_path and labels_path:
        debug_print(f"Model found locally: {model_path}")
        debug_print(f"Labels found: {labels_path}")
        return str(model_path), str(labels_path)

    # 1) Try to get file id from Streamlit secrets (recommended) or environment var
    file_id = None
    # Preferred: st.secrets (set via Streamlit app settings)
    try:
        file_id = st.secrets.get("MODEL_GDRIVE_ID")
    except Exception:
        file_id = None
    # Fallback: environment variable (if you set it that way)
    if not file_id:
        file_id = os.environ.get("MODEL_GDRIVE_ID")

    debug_print(f"st.secrets MODEL_GDRIVE_ID: {st.secrets.get('MODEL_GDRIVE_ID')}")
    debug_print(f"os.environ MODEL_GDRIVE_ID: {os.environ.get('MODEL_GDRIVE_ID')}")

    if not file_id:
        raise RuntimeError(
            "Model not found in repo and no MODEL_GDRIVE_ID set. "
            "Please either commit a 'models/' folder (with final_saved_model and labels.json) "
            "to your repo OR upload a zip of 'models/' to Google Drive and set MODEL_GDRIVE_ID in Streamlit Secrets."
        )

    # 2) Download & extract
    download_and_extract_from_drive(file_id)

    # 3) Re-check for model and labels
    model_path, labels_path = find_model_path()
    if not model_path:
        # List top-level models directory contents for debug
        contents = list(MODELS_ROOT.iterdir()) if MODELS_ROOT.exists() else []
        raise RuntimeError(
            "Downloaded model but could not find a usable model file/folder.\n"
            f"Top-level models/ contents: {[p.name for p in contents]}\n"
            "Make sure your zip, when unzipped, creates a 'models' folder that contains "
            "'final_saved_model' (SavedModel dir) or a '.keras'/.h5 file and labels.json."
        )
    if not labels_path:
        raise RuntimeError(
            f"Model found at {model_path} but labels.json not found under models/. "
            "Ensure your zip contains models/labels.json or commit labels.json to repo."
        )

    debug_print(f"Model available at: {model_path}")
    debug_print(f"Labels available at: {labels_path}")
    return str(model_path), str(labels_path)

@st.cache_resource
def load_model_and_labels():
    model_path, labels_path = ensure_model_available()
    # Load model: tf.keras.load_model accepts both a dir (SavedModel) or a file (.keras/.h5)
    debug_print(f"Loading model from {model_path} ...")
    model = tf.keras.models.load_model(model_path)
    debug_print("Model loaded. Loading labels...")
    with open(labels_path, "r") as f:
        labels = json.load(f)
    debug_print("Labels loaded.")
    return model, labels

def main():
    st.title("Image Classifier Demo (Streamlit Cloud)")
    # Show helpful instructions / debug info to the user
    st.markdown(
        "App will try to find `models/` in the repo first. If missing, it will download a zip from Google Drive "
        "using the `MODEL_GDRIVE_ID` secret (set in Streamlit Cloud Secrets)."
    )

    try:
        model, labels = load_model_and_labels()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        # show extra debugging help
        st.info("Checklist:\n"
                "- Did you set MODEL_GDRIVE_ID in Streamlit Secrets (Settings → Secrets)?\n"
                "- Is the Drive file shared as 'Anyone with the link' (Viewer)?\n"
                "- Does the zip contain a top-level 'models' folder with final_saved_model and labels.json?")
        return

    uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB").resize((224,224))
        st.image(img, caption="Uploaded image", use_column_width=True)
        x = np.array(img)/255.0
        x = np.expand_dims(x, axis=0)
        preds = model.predict(x)[0]
        top_idx = int(preds.argmax())
        st.write(f"Prediction: {labels[top_idx]} ({preds[top_idx]*100:.2f}%)")

if __name__ == "__main__":
    main()
