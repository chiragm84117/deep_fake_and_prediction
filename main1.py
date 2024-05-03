import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.applications.xception import Xception, preprocess_input as preprocess_input_xception, decode_predictions as decode_predictions_xception

# Function to load and preprocess the image
def load_and_preprocess_image(image_path, model_name):
    img = keras_image.load_img(image_path, target_size=(224, 224))
    if model_name == 'VGG16':
        img = preprocess_input(keras_image.img_to_array(img))
    elif model_name == 'Xception':
        img = preprocess_input_xception(keras_image.img_to_array(img))
    return img

# Function to make predictions
def predict(image, model_name):
    if model_name == 'VGG16':
        model = VGG16()
        img = preprocess_input(image)
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        preds = model.predict(img)
        label = decode_predictions(preds)
    elif model_name == 'Xception':
        model = Xception()
        img = preprocess_input_xception(image)
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        preds = model.predict(img)
        label = decode_predictions_xception(preds)
    return label

# Streamlit App
st.title('Image Classification')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    model_option = st.selectbox("Select Model", ["VGG16", "Xception"])

    if st.button('Classify'):
        if model_option == "VGG16":
            image = load_and_preprocess_image(uploaded_file, 'VGG16')
            label = predict(image, 'VGG16')
            st.write('Prediction using VGG16 model:')
            st.write(label[0][0][1])
        elif model_option == "Xception":
            image = load_and_preprocess_image(uploaded_file, 'Xception')
            label = predict(image, 'Xception')
            st.write('Prediction using Xception model:')
            st.write(label[0][0][1])
