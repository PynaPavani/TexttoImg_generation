import streamlit as st
from pathlib import Path
import tqdm
import torch
import pandas as pd
import numpy as np
from diffusers import StableDiffusionPipeline
from transformers import pipeline, set_seed
import matplotlib.pyplot as plt
import cv2


st.title("Text to Image generation")
st.header("Generate Image with Text Description using Stable Diffusion")

model_id = "stabilityai/stable-diffusion-2"  
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Stable Diffusion pipeline with appropriate settings
image_gen_model = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    variant="fp16",
    
)

# Move the model to the specified device
image_gen_model = image_gen_model.to(device)

# Set guidance scale if needed
image_gen_model.scheduler.guidance_scale = 9
# Define the text prompt for image generation
text=st.text_input('enter the text that you want to generate image')

if st.button('generate image'):
    text_prompt = text
    with torch.autocast("cuda"):
      generated_images = image_gen_model(text_prompt).images
      # Display the generated image
      st.image((generated_images[0]),caption='generated image',width='40px')
      plt.show()
      
