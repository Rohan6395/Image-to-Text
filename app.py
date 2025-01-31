import streamlit as st
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# Load Pretrained Model
model_name = "nlpconnect/vit-gpt2-image-captioning"
model = VisionEncoderDecoderModel.from_pretrained(model_name)
processor = ViTImageProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


import numpy as np
# Function to generate captions
def generate_caption(image):
    image = image.convert("RGB")  # Convert to RGB mode
    image = np.array(image)  # Convert to NumPy array
    image = processor(images=image, return_tensors="pt").pixel_values
    output_ids = model.generate(image, max_length=16, num_beams=4)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

# Streamlit UI
st.set_page_config(page_title="Image Caption Generator", page_icon="🖼", layout="wide")

st.title("🖼 Image Caption Generator")
st.write("Upload an image, and the Model will generate a caption for it!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Generate Caption Button
    if st.button("Generate Caption ✨"):
        with st.spinner("Generating caption... ⏳"):
            caption = generate_caption(image)
            st.success("Caption Generated! ✅")
            st.write(f"**Caption:** {caption}")

# Footer
st.markdown("---")
st.write("Developed by **Rohan Sharma**")

