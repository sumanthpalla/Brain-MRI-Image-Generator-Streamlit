import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import io
import os

# Set up the Streamlit app
st.title("Brain MRI Generator")

# Load the diffusion model
@st.cache_resource
def load_model(use_finetuned=False):
    if use_finetuned and os.path.exists("finetuned_model"):
        model_path = "finetuned_model"
    else:
        model_path = "Xuanyuan/MRI_brain_tumor_seg"
    
    pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

# Checkbox to use finetuned model
use_finetuned = st.checkbox("Use finetuned model", value=False)

pipe = load_model(use_finetuned)

# File uploader for the original image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the original image
    original_image = Image.open(uploaded_file)
    st.image(original_image, caption="Original Image", use_column_width=True)
    
    # Generate button
    if st.button("Generate Brain MRI"):
        with st.spinner("Generating Brain MRI..."):
            # Prepare the image for the model
            input_image = original_image.resize((512, 512))
            input_image = input_image.convert("RGB")
            
            # Generate the Brain MRI
            output_image = pipe(image=input_image, strength=0.75, guidance_scale=7.5).images[0]
            
            # Display the generated image
            st.image(output_image, caption="Generated Brain MRI", use_column_width=True)
            
            # Provide download button for the generated image
            buf = io.BytesIO()
            output_image.save(buf, format="PNG")
            btn = st.download_button(
                label="Download Generated MRI",
                data=buf.getvalue(),
                file_name="generated_brain_mri.png",
                mime="image/png"
            )

st.write("Note: This app uses a pre-trained model to generate brain MRI images. The results are for demonstration purposes only and should not be used for medical diagnosis.")
