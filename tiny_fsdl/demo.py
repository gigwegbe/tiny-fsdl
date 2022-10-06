import os
from pathlib import Path
import tempfile


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import preprocess

st.title("Tiny-FSDL Demo")
st.write("By George Igwegbe and David Rose")

st.write("## Preprocessing")
st.write("This is a demo of the preprocessing module.")

# Select an image to load
input_dir = st.selectbox(
    "Select an folder",
    (
        "data1",
        "data2",
        "data3",
        "data4",
        "data5",
    ),
)
input_dir = "../ml/data/raw_images/" + input_dir
output_dir = Path(input_dir).parents[1] / "processed_streamlit" / Path(input_dir).name
st.write("You selected:", input_dir)

image_file_list = os.listdir(input_dir)
image_file = st.selectbox("Select file", image_file_list)
st.write("You selected:", image_file)

# Load the image in streamlit
image_path = Path(input_dir, image_file)
st.image(str(image_path), use_column_width=True)

# Preprocess the image in opencv for cropping
pre = preprocess.Preprocess()
pre.process_image(
    image_path,
    output_dir,
)

# Load the cropped image in streamlit
cropped_paths = os.listdir(output_dir)
st.write("Cropped images:", cropped_paths)
for fp in cropped_paths:
    st.image(str(Path(output_dir, fp)), width=80)
