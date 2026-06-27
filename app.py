import streamlit as st
import tensorflow as tf
import numpy as np
import os
import requests
from tensorflow.keras.utils import load_img, img_to_array
import json
 
st.set_page_config(page_title="Crop Disease Detection AI", page_icon="🌱", layout="centered")
 
# ---------------- DISEASE KNOWLEDGE BASE ----------------
DISEASE_DB = {
    "capsicum_Healthy": {
        "category": "Healthy", "emoji": "🌱",
        "display_name": "Healthy Capsicum Leaf", "type": "Healthy",
        "description": "No visible signs of disease or stress on the leaf.",
        "causes": "N/A — plant is healthy.", "spread": "N/A",
        "precautions": ["Continue routine monitoring", "Maintain balanced watering and fertilization"],
        "organic_treatment": ["No treatment needed"],
        "chemical_treatment": ["No treatment needed"],
        "prevention": ["Regular inspection", "Proper spacing for airflow", "Balanced NPK fertilization"]
    },
    "capsicum_Bacterial_spot": {
        "category": "Bacterial disease", "emoji": "🦠",
        "display_name": "Bacterial Spot (Capsicum)", "type": "Bacterial",
        "description": "Small, dark, water-soaked lesions on leaves that turn brown and may cause leaf drop.",
        "causes": "Xanthomonas campestris pv. vesicatoria bacteria.",
        "spread": "Splashing water, rain, contaminated tools/seeds; favored by warm, humid, wet weather.",
        "precautions": ["Remove and destroy infected leaves", "Avoid overhead irrigation", "Don't work fields when leaves are wet"],
        "organic_treatment": ["Copper-based organic bactericide spray", "Neem oil to limit secondary infection"],
        "chemical_treatment": ["Copper oxychloride or streptomycin-based bactericides (follow local label dosage)"],
        "prevention": ["Use certified disease-free seeds", "2–3 year crop rotation", "Improve spacing/drainage", "Disinfect tools between plants"]
    },
    "cucumber_Healthy_leaves": {
        "category": "Healthy", "emoji": "🌱",
        "display_name": "Healthy Cucumber Leaf", "type": "Healthy",
        "description": "No visible signs of disease or stress on the leaf.",
        "causes": "N/A — plant is healthy.", "spread": "N/A",
        "precautions": ["Continue routine monitoring", "Maintain balanced watering and fertilization"],
        "organic_treatment": ["No treatment needed"],
        "chemical_treatment": ["No treatment needed"],
        "prevention": ["Regular inspection", "Proper spacing for airflow", "Balanced NPK fertilization"]
    },
    "cucumber_Downy_mildew": {
        "category": "Fungal disease", "emoji": "🍄",
        "display_name": "Downy Mildew (Cucumber)", "type": "Fungal (oomycete)",
        "description": "Yellow angular spots on top of leaf, greyish-purple fuzz underneath; leaves curl and dry.",
        "causes": "Pseudoperonospora cubensis (oomycete pathogen).",
        "spread": "Wind-blown spores in cool, humid, wet conditions; worsened by overhead watering.",
        "precautions": ["Remove infected leaves", "Prune for airflow", "Water at the base, not the leaves"],
        "organic_treatment": ["Copper-based fungicide spray", "Baking soda + neem oil mix"],
        "chemical_treatment": ["Mancozeb or chlorothalonil-based fungicides", "Metalaxyl-based systemic fungicide for severe cases"],
        "prevention": ["Use resistant varieties", "Drip irrigation instead of overhead", "Good drainage/spacing", "Preventive spray during humid seasons"]
    },
    "cucumber_Powdery_mildew": {
        "category": "Fungal disease", "emoji": "🍄",
        "display_name": "Powdery Mildew (Cucumber)", "type": "Fungal",
        "description": "White powdery growth on leaves/stems; leaves yellow and turn brittle.",
        "causes": "Podosphaera xanthii or Erysiphe cichoracearum fungi.",
        "spread": "Airborne spores; favored by warm days, cool nights, high humidity, poor airflow.",
        "precautions": ["Remove heavily infected leaves", "Avoid excess nitrogen", "Increase plant spacing"],
        "organic_treatment": ["Sulfur-based fungicide spray", "1:10 diluted milk spray", "Neem oil spray"],
        "chemical_treatment": ["Triazole fungicides (myclobutanil, tebuconazole)", "Strobilurin fungicides for resistant cases"],
        "prevention": ["Use resistant varieties", "Prune for spacing/airflow", "Avoid excess nitrogen", "Monitor in warm-day/cool-night weather"]
    },
}
 
DEFAULT_INFO = {
    "category": "Unknown", "emoji": "❓", "display_name": "Unrecognized class", "type": "Unknown",
    "description": "No information available for this class yet.",
    "causes": "N/A", "spread": "N/A",
    "precautions": ["Consult a local agricultural expert"],
    "organic_treatment": ["Consult a local agricultural expert"],
    "chemical_treatment": ["Consult a local agricultural expert"],
    "prevention": ["Consult a local agricultural expert"]
}
 
# ---------------- MODEL DOWNLOAD + LOAD ----------------
# 👇 हे YOUR GitHub Release ची लिंक टाका (Releases page वरून "Copy link address" केलेली)
MODEL_URL = "https://github.com/atharvghodake248-png/-Crop-Disease-Detection-System/releases/download/v1.0/crop_model.keras"
MODEL_PATH = "crop_model.keras"
 
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model... (only happens once)"):
            response = requests.get(MODEL_URL)
            response.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
    return tf.keras.models.load_model(MODEL_PATH, compile=False)
 
model = load_model()
 
with open("class_names.json", "r") as f:
    class_indices = json.load(f)
class_names = {v: k for k, v in class_indices.items()}
 
IMG_SIZE = (224, 224)
 
st.title("🌱 Crop Disease Detection AI")
st.write("Upload a leaf image to detect disease, see treatment options, and check prediction confidence.")
 
uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "png", "jpeg"])
 
def predict(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return model.predict(img_array)[0]  # full probability vector
 
if uploaded_file is not None:
    file_path = "temp.jpg"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
 
    st.image(uploaded_file, caption="Uploaded Image")
 
    probs = predict(file_path)
    top_idx = int(np.argmax(probs))
    top_class = class_names[top_idx]
    top_conf = float(probs[top_idx])
    info = DISEASE_DB.get(top_class, DEFAULT_INFO)
 
    # 1. SMART HEALTH CLASSIFICATION
    st.markdown(f"## {info['emoji']} {info['category']}")
    st.success(f"**Prediction:** {info['display_name']}  \n**Confidence:** {top_conf*100:.1f}%")
 
    # 4. CONFIDENCE SCORE DISPLAY
    st.subheader("📊 Confidence Breakdown — Top 3")
    top3 = np.argsort(probs)[::-1][:3]
    for idx in top3:
        cname = class_names[idx]
        cinfo = DISEASE_DB.get(cname, DEFAULT_INFO)
        pct = float(probs[idx]) * 100
        st.write(f"{cinfo['emoji']} **{cinfo['display_name']}** — {pct:.1f}%")
        st.progress(min(int(pct), 100))
 
    # 2. DISEASE INFORMATION PANEL
    st.subheader("📋 Disease Information")
    st.write(f"**Disease Name:** {info['display_name']}")
    st.write(f"**Type:** {info['type']}")
    st.write(f"**Description:** {info['description']}")
    st.write(f"**Causes:** {info['causes']}")
    st.write(f"**Spread Conditions:** {info['spread']}")
 
    # 3. PRECAUTION & TREATMENT SYSTEM
    if info["category"] != "Healthy":
        st.subheader("🛡️ Precautions & Treatment")
        with st.expander("⚠️ Immediate Precautions"):
            for item in info["precautions"]:
                st.write(f"- {item}")
        with st.expander("🌿 Organic Treatment"):
            for item in info["organic_treatment"]:
                st.write(f"- {item}")
        with st.expander("🧪 Chemical Treatment"):
            for item in info["chemical_treatment"]:
                st.write(f"- {item}")
        with st.expander("🚜 Prevention Steps for Farmers"):
            for item in info["prevention"]:
                st.write(f"- {item}")
        st.caption("⚠️ Treatment info is general guidance only — confirm products and dosages with your local agricultural extension office before applying chemicals.")
    else:
        st.balloons()
        st.info("✅ This plant looks healthy! Keep up the good care.")
 