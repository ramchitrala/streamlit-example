import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array

# Load the ResNet50 model pre-trained on ImageNet data
model = ResNet50(weights='imagenet')

st.title('Image Classification with ResNet50')

st.write("This app uses a pre-trained ResNet50 model to classify images. Upload an image, and see the predictions.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    # Convert the image to a numpy array and preprocess it
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Make predictions
    predictions = model.predict(img_array)
    # Convert predictions into readable labels
    labels = decode_predictions(predictions)
    
    # Show the top 3 predictions
    for label in labels[0][:3]:
        st.write(f"{label[1]} ({label[2]*100:.2f}%)")
