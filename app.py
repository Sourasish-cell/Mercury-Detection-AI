import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

from model import load_model

# -------------------------
# Load classes from file
# -------------------------
def load_classes(file_path):
    with open(file_path, "r") as f:
        return [line.strip() for line in f.readlines()]


st.title("AI-Based Heavy Metal Detection System")

st.write("""
Upload a fluorescence image and select the metal type.
The model predicts concentration based on fluorescence quenching.
""")


# -------------------------
# Metal selection
# -------------------------
metal = st.selectbox(
    "Select Metal",
    ["Mercury (Hg)", "Zinc (Zn)"]
)


# -------------------------
# Load model + classes
# -------------------------
if metal == "Mercury (Hg)":
    model = load_model("best_mercury_model1.pth")
    class_names = load_classes("mercury_classes.txt")

else:
    model = load_model("best_zinc_model1.pth")
    class_names = load_classes("classes.txt")


# -------------------------
# Upload image
# -------------------------
uploaded_file = st.file_uploader(
    "Upload Fluorescence Image",
    type=["jpg", "png", "jpeg"]
)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


# -------------------------
# Prediction
# -------------------------
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)

        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    st.success(f"Predicted: **{class_names[pred]}**")
    st.info(f"Confidence: {confidence * 100:.2f}%")

    st.subheader("Prediction Probabilities")

    for i, cls in enumerate(class_names):
        st.write(f"{cls}: {probs[0][i].item():.2f}")
