import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# --- Config ---
MODEL_PATH = "best_densenet121.pth"
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="Pneumonia CXR Classifier", layout="centered")


@st.cache_resource
def load_model():
    model = models.densenet121(weights=None)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(num_ftrs, 1)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model


eval_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict(image: Image.Image, model: nn.Module) -> float:
    image = image.convert("RGB")  # Matches ImageFolder's 3-channel loading
    tensor = eval_transforms(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logit = model(tensor)
        prob = torch.sigmoid(logit).item()
    return prob


# --- UI ---
st.title("Pneumonia CXR Classifier")
st.write("Upload a Chest X-Ray image to test the trained DenseNet-121 model.")

threshold = st.sidebar.slider("Classification Threshold", 0.1, 0.9, 0.5, 0.05)

uploaded_file = st.file_uploader("Choose a CXR image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Chest X-Ray", use_container_width=True)

    model = load_model()
    prob_pneumonia = predict(image, model)
    is_pneumonia = prob_pneumonia >= threshold

    st.divider()
    if is_pneumonia:
        st.error(f"**Prediction: PNEUMONIA** (Confidence: {prob_pneumonia * 100:.2f}%)")
    else:
        st.success(f"**Prediction: NORMAL** (Confidence: {(1 - prob_pneumonia) * 100:.2f}%)")

    st.progress(prob_pneumonia, text=f"Pneumonia Probability: {prob_pneumonia:.4f}")