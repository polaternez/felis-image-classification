import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, request, render_template
import tensorflow as tf
from keras.models import load_model
from PIL import Image



# Constants
IMG_SIZE = (224, 224)
CLASS_NAMES = [
    'african-wildcat', 'blackfoot-cat', 'chinese-mountain-cat',
    'domestic-cat', 'european-wildcat', 'jungle-cat', 'sand-cat'
]

# Load pre-trained model
model = load_model("models/felis_taxonomy_efficientnet.keras")

def preprocess_image(image):
    """
    Pre-processes the input image for model prediction.
    Resizes and converts the image to a format suitable for model input.

    Args:
        image (FileStorage): Image file uploaded via HTTP request.

    Returns:
        numpy.ndarray: Pre-processed image ready for prediction.
    """
    img = Image.open(image)
    img = img.resize(IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    return img_array

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    """
    Renders the home page.

    Returns:
        str: HTML content for the home page.
    """
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to predict the class of an uploaded image.

    Returns:
        str: HTML content with the prediction result.
    """
    image_file = request.files['imageInput']
    
    # Pre-process the image
    preprocessed_image = preprocess_image(image_file)
    
    # Make prediction
    prediction = model.predict(preprocessed_image)
    predicted_class = CLASS_NAMES[tf.argmax(prediction[0])]
    
    return render_template("index.html", predicted_class=predicted_class)

if __name__ == "__main__":
    app.run(debug=True)
