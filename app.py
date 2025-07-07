import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import cv2
import time
import os

st.set_page_config(page_title="Thermal Image Cropper", layout="centered")
st.title("ThermalScope: Microscopic Region Cropper")
uploaded_file = st.file_uploader("Upload your thermal image", type=["png", "jpg", "jpeg", "tiff"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.markdown("Uploaded Image:")
    st.image(img, caption="Original Image", use_column_width=True)

    w, h = img.size

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=3,
        stroke_color="black",
        background_image=img,
        update_streamlit=True,
        height=h,
        width=w,
        drawing_mode="rect",
        key="canvas",
    )
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        if objects:
            obj = objects[0]
            x = int(obj["left"])
            y = int(obj["top"])
            width = int(obj["width"])
            height = int(obj["height"])
            img_np = np.array(img)
            cropped = img_np[y:y+height, x:x+width]

            st.markdown("Cropped Output:")
            st.image(cropped, caption="Zoomed-in Region", use_column_width=False)
            save_dir = "data/cropped_outputs"
            os.makedirs(save_dir, exist_ok=True)
            filename = f"cropped_{int(time.time())}.png"
            cv2.imwrite(os.path.join(save_dir, filename), cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
            st.success(f"Cropped image saved as {filename}")