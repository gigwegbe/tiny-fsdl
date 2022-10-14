import os
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

import preprocess


def infer(model_path, img):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]["index"])
    return np.argmax(prediction[0])


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
bounded_img = pre.process_image(
    image_path,
    output_dir,
)

# Display the image with bounding boxes
st.write("## Bounding Boxes")
st.image(bounded_img, use_column_width=True)

# Run Inference on the cropped images
cropped_paths = os.listdir(output_dir)
predictions = []
for fp in cropped_paths:
    img = cv2.imread(str(Path(output_dir, fp))).astype(np.int8)
    img = np.expand_dims(img, axis=0)
    img = np.swapaxes(img, 1, 2)
    prediction = infer("../ml/ml/models/digit_model_quantized.tflite", img)
    predictions.append(str(prediction))

st.write("## Inference")
st.write("The model predicts the following digits:")
st.write(f"### {''.join(predictions)}")
