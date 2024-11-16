# import streamlit as st
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np
# from PIL import Image
# import tensorflow as tf

# # Display TensorFlow version (optional, for debugging)
# # st.write(f"Using TensorFlow version: {tf.__version__}")

# # Load the pre-trained .h5 model
# try:
#     model = load_model('D:\\Python For Data Science Course\\Python\\01_streamlit\\07_obj_detect\\model.h5')  # Update to your model path
# except Exception as e:
#     st.error(f"Failed to load the model: {e}")
#     st.stop()

# # Define the class names (CIFAR-10 classes in this case)
# class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# # Streamlit app title
# st.title("Image Recognition using CNN")

# # Upload an image file
# uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# if uploaded_file is not None:
#     # Convert the file to an image
#     img = Image.open(uploaded_file)
#     st.image(img, caption="Uploaded Image", use_container_width=True)
#     st.write("Classifying...")

#     # Preprocess the image
#     img = img.resize((32, 32))  # Resize to 32x32 for CIFAR-10 compatibility
#     img_array = image.img_to_array(img) / 255.0  # Normalize the image
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

#     # Make a prediction
#     predictions = model.predict(img_array)
#     predicted_class = np.argmax(predictions)
#     predicted_class_name = class_names[predicted_class]

#     # Display the prediction result
#     st.write(f"Predicted class: **{predicted_class_name}**")
#     print(predicted_class_name)


import os

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Load the model
try:
    model = load_model('D:\\Python For Data Science Course\\Python\\01_streamlit\\07_obj_detect\\model.h5')
except Exception as e:
    st.error(f"Failed to load the model: {e}")
    st.stop()

# Define the class names (CIFAR-10 classes in this case)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# App title and description
st.title("Advanced Image Recognition using CNN")
st.markdown("This app classifies images using a Convolutional Neural Network (CNN) model. Upload an image, and the app will predict its class.")

# Upload an image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Prediction confidence threshold
confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5)

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocessing options
    st.sidebar.title("Image Augmentation Options")
    if st.sidebar.checkbox("Flip Image Horizontally"):
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    rotation_angle = st.sidebar.slider("Rotate Image", -45, 45, 0)
    if rotation_angle != 0:
        img = img.rotate(rotation_angle)
    
    st.write("Classifying...")
    
    # Preprocess the image
    img = img.resize((32, 32))
    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make a prediction
    predictions = model.predict(img_array)[0]
    predicted_class = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class]
    confidence_score = predictions[predicted_class]
    
    # Display prediction result and confidence
    if confidence_score < confidence_threshold:
        st.write(f"Prediction confidence is below the threshold of {confidence_threshold:.2f}.")
    else:
        st.write(f"Predicted class: **{predicted_class_name}** with confidence **{confidence_score:.2f}**")
    
    # Display a bar chart of all prediction probabilities
    st.subheader("Prediction Confidence Scores")
    fig, ax = plt.subplots()
    ax.barh(class_names, predictions)
    ax.set_xlabel("Confidence Score")
    ax.set_xlim(0, 1)
    st.pyplot(fig)

    # Display TensorFlow and hardware info
    st.sidebar.write("**TensorFlow Version:**", tf.__version__)
    st.sidebar.write("**Device Information:**", tf.config.list_physical_devices())
else:
    st.write("Please upload an image to start the classification.")
