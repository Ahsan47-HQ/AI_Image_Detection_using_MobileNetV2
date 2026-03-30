import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model("model/best_model.h5")

def preprocess(image):
    image = image.resize((224,224))
    image = np.array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = np.expand_dims(image,axis=0)
    return image

def predict(image):
    img = preprocess(image)
    preds = model.predict(img)
    class_idx = np.argmax(preds)
    confidence = float(np.max(preds))
    
    label = "REAL" if class_idx == 1 else "FAKE"
    return label,confidence