import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------
# Load the trained models (your paths)
# ---------------------------
cnn_model = tf.keras.models.load_model(
    r"C:\Users\hp\OneDrive\Desktop\Projects\Project 2 (Cancer Detection Using Histopathological Images)\models\lung_cancer_cnn.h5"
)
vgg16_model = tf.keras.models.load_model(
    r"C:\Users\hp\OneDrive\Desktop\Projects\Project 2 (Cancer Detection Using Histopathological Images)\models\lung_cancer_vgg16.h5"
)

# ---------------------------
# Define class labels (update if your dataset has different categories)
# ---------------------------
CLASS_NAMES = ["Adenocarcinoma", "Large Cell Carcinoma", "Squamous Cell Carcinoma"]

# ---------------------------
# Prediction Function
# ---------------------------
def predict(image, model_choice):
    # Resize and preprocess image
    img = image.convert("RGB").resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Choose model
    if model_choice == "CNN":
        prediction = cnn_model.predict(img_array)
    else:
        prediction = vgg16_model.predict(img_array)

    # Get prediction result
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    return {predicted_class: confidence}

# ---------------------------
# Gradio Interface
# ---------------------------
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil"),
        gr.Radio(["CNN", "VGG16"], label="Choose Model", value="CNN")
    ],
    outputs=gr.Label(num_top_classes=3),
    title="🫁 Lung Cancer Detection",
    description="Upload a histopathological lung image and the model will predict the cancer type."
)

iface.launch()
