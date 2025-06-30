import numpy as np
import streamlit as st
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow_hub.keras_layer import KerasLayer 
import tensorflow as tf
from tensorflow.keras.models import load_model


def run():
    #load model
    file = st.file_uploader("Upload an image", type=["jpg", "png"])

    model = load_model('model_terbaik.keras', custom_objects={'KerasLayer': KerasLayer})

    def import_and_predict(image_data, model):
        image = load_img(image_data, target_size=(224, 224))
        img_array = img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        # Normalize the image
        img_array = img_array / 255.0

        # Make prediction
        predictions = model.predict(img_array)

        # Get the class with the highest probability
        idx = np.where(predictions >= 0.5, 1, 0).item()
        # predicted_class = np.argmax(predictions)

        jenis = ['Kebakaran', 'Tidak Kebakaran']
        result = f"Prediction: {jenis[idx]}"

        return result

    if file is None:
        st.text("Please upload an image file")
    else:
        result = import_and_predict(file, model)
        st.image(file)
        st.write(result)

if __name__ == '__main__':
    run()