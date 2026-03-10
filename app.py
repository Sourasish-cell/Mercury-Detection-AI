import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from model import load_model

# Page settings
st.set_page_config(page_title="Mercury Detection AI", layout="centered")

st.title("AI Based Mercury Concentration Detection")
st.write("Upload a fluorescence image to predict Mercury concentration.")

# Load model
model = load_model("best_mercury_model1.pth")

# Load class labels
with open("classes.txt") as f:
    classes = [line.strip() for line in f.readlines()]

# Image transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

uploaded_file = st.file_uploader("Upload Fluorescent Image", type=["png","jpg","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = transform(image).unsqueeze(0)

    with torch.no_grad():

        outputs = model(img)
        _, pred = torch.max(outputs,1)

        concentration = classes[pred.item()]

    
    prob = torch.nn.functional.softmax(outputs, dim=1)
    confidence = prob[0][pred.item()].item()*100

    st.success(f"Predicted Mercury Concentration: {concentration}")
    st.write(f"Confidence: {confidence:.2f}%")
