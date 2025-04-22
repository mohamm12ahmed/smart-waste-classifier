import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import numpy as np
from model import loaded_model

# تحديد الجهاز
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# تحميل الموديل
model = loaded_model.to(device)
model.load_state_dict(torch.load("waste_classifier.pt", map_location=device))
model.eval()

# الفئات
classes = ["Plastic", "Glass", "Paper", "Cardboard", "Metal", "Trash"]

# تحويل الصورة
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# واجهة التطبيق
st.title("📱 Smart Waste Classification System")
st.subheader("Choose an image input method:")

input_option = st.radio("Select input method:", ["📷 Capture with Camera", "🖼️ Upload from Device", "🎥 Live Video Detection"])

image = None

# 📷 الكاميرا
if input_option == "📷 Capture with Camera":
    camera_image = st.camera_input("Take a picture")
    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")

# 🖼️ رفع صورة
elif input_option == "🖼️ Upload from Device":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

# 🎥 فيديو مباشر
elif input_option == "🎥 Live Video Detection":
    stframe = st.empty()
    cap = cv2.VideoCapture(0)

    run = st.checkbox("Start Camera")

    while run and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame.")
            break

        # تحويل الصورة إلى RGB لـ PIL
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img).convert("RGB")

        # تجهيز الصورة للتصنيف
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_class = classes[predicted.item()]

        # كتابة التصنيف على الصورة
        cv2.putText(frame, f"Class: {predicted_class}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # عرض الصورة في Streamlit
        stframe.image(frame, channels="BGR")

    cap.release()
    st.warning("Video stream ended.")

# تصنيف صورة عادية
if image is not None:
    st.image(image, caption="📸 Input Image", use_column_width=True)

    if st.button("Classify"):
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_class = classes[predicted.item()]

        st.success(f"✅ Detected category: **{predicted_class}**")

        image_path = f"images/{predicted_class.lower()}.png"
        try:
            st.image(image_path, caption=f"🗑️ {predicted_class} bin", width=200)
        except:
            st.info("No illustrative image available for this category.")
