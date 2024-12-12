
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model


# Recreate base_model for feature extraction
img_height = 180
img_width = 180
categories = ['freshapples', 'rottenapples', 'freshbanana', 'rottenbanana']



# In[3]:



# Classification function using the loaded model
def classify_image(image_path, feature_extractor, loaded_model):
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)  # Preprocess for MobileNetV2
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch

    # Extract features
    features = feature_extractor.predict(img_array)
    features = features.reshape(1, -1)  # Flatten features
    
    # Predict using the loaded model
    prediction = loaded_model.predict(features)
    predicted_class = np.argmax(prediction)
    print(f"Prediction: {categories[predicted_class]}")
    return categories[predicted_class]




# In[ ]:



