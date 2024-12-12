#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy
import matplotlib.pyplot as plt

# In[2]:


# Ensure TensorFlow uses GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("GPU available:", tf.config.experimental.get_device_details(physical_devices[0])['device_name'])
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
else:
    print("No GPU found. Using CPU instead.")

# In[3]:


# Visualization function
def visualize_sample_images(dataset_dir, categories):
    n = len(categories)
    fig, axs = plt.subplots(1, n, figsize=(20, 5))
    for i, category in enumerate(categories):
        folder = os.path.join(dataset_dir, category)
        image_file = os.listdir(folder)[0]
        img_path = os.path.join(folder, image_file)
        img = load_img(img_path)
        axs[i].imshow(img)
        axs[i].set_title(category)
    plt.tight_layout()
    plt.show()

# In[4]:


# Directories
dataset_base_dir = 'E:/ML_Proj/Dataset'  # Change this to your actual dataset path
train_dir = os.path.join(dataset_base_dir, 'Train')
categories = ['freshapples', 'rottenapples', 'freshbanana', 'rottenbanana']
visualize_sample_images(train_dir, categories)

# In[5]:


# Parameters
img_height = 180
img_width = 180
batch_size = 32

# In[6]:


# Load dataset and extract labels
def load_data_and_labels(directory, categories):
    data = []
    labels = []
    for i, category in enumerate(categories):
        folder = os.path.join(directory, category)
        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)
            img = load_img(img_path, target_size=(img_height, img_width))
            img_array = img_to_array(img)
            img_array = preprocess_input(img_array)  # Preprocess for MobileNetV2
            data.append(img_array)
            labels.append(i)  # Assign numeric label for the category
    return np.array(data), np.array(labels)

# In[7]:


# Load data
X, y = load_data_and_labels(train_dir, categories)

# One-hot encode labels for multi-class classification
from tensorflow.keras.utils import to_categorical
y = to_categorical(y, num_classes=len(categories))

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1))

# Feature extraction using MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
X_train_features = base_model.predict(X_train, batch_size=batch_size)
X_test_features = base_model.predict(X_test, batch_size=batch_size)

# Flatten features for input into the neural network
X_train_features = X_train_features.reshape(X_train_features.shape[0], -1)
X_test_features = X_test_features.reshape(X_test_features.shape[0], -1)

# In[8]:


# Define neural network model
model = Sequential([
    Input(shape=(X_train_features.shape[1],)),
    Dense(128, activation='relu'),  # Hidden layer
    Dense(len(categories), activation='softmax')  # Output layer
])

# In[9]:


# Compile the model
learning_rate = 0.001
model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss='categorical_crossentropy',
    metrics=[CategoricalAccuracy()]
)

# In[10]:


# Train the model
epochs = 50
history = model.fit(
    X_train_features, y_train,
    validation_data=(X_test_features, y_test),
    epochs=epochs,
    batch_size=batch_size
)

# In[11]:


# Evaluate the model
eval_result = model.evaluate(X_test_features, y_test)
print(f"Test Loss: {eval_result[0]}, Test Accuracy: {eval_result[1] * 100:.2f}%")

# In[12]:


# Classification function
def classify_image(image_path, feature_extractor, model):
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)  # Preprocess for MobileNetV2
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch

    features = feature_extractor.predict(img_array)
    features = features.reshape(1, -1)  # Flatten features
    prediction = model.predict(features)
    predicted_class = np.argmax(prediction)
    print(f"Prediction: {categories[predicted_class]}")

# In[13]:


# Example usage
image_path = 'E:/ML_Proj/Dataset/Train/freshapples/a_f028.png'  # Change this to the path of your image
classify_image(image_path, base_model, model)

# In[ ]:



