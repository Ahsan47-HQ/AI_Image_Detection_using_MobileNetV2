import numpy as np
from PIL import Image
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model/model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

def predict(image):
    img = preprocess_image(image)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img)

    # Run inference
    interpreter.invoke()

    # Get output
    preds = interpreter.get_tensor(output_details[0]['index'])

    class_idx = np.argmax(preds)
    confidence = float(np.max(preds))

    label = "REAL" if class_idx == 1 else "FAKE"

    return label, confidence