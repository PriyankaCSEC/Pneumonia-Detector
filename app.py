import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model('model/pneumonia_model.h5')

def preprocess(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

st.title("Pneumonia Detection from Chest X-rays : ")
uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # filename = f"{timestamp}_{uploaded_file.name}"
    # file_path = os.path.join(uploads, filename)



    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed = preprocess(image)
    prediction = model.predict(processed)[0][0]

    if prediction < 0.5:
        st.error(f"Prediction: NORMAL ({(1 - prediction)*100:.2f}%)")
    else:
        st.warning(f"Prediction: PNEUMONIA ({prediction*100:.2f}%)")
