import os 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, request, render_template
import tensorflow as tf
from keras.models import load_model
from PIL import Image


IMG_SIZE = (224, 224)
class_names = [
    'african-wildcat', 'blackfoot-cat', 'chinese-mountain-cat',
    'domestic-cat', 'european-wildcat', 'jungle-cat', 'sand-cat'
]

# Load model
model = load_model("models/felis_taxonomy_efficientnet.keras")

# Function to pre-process the image
def preprocess_image(img):
    img = Image.open(img)
    img = img.resize(IMG_SIZE)
    img_arr = tf.keras.preprocessing.image.img_to_array(img)
    img_arr = img_arr.reshape((1, IMG_SIZE[0], IMG_SIZE[1], 3))

    return img_arr


# WSGI
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

# API endpoint to predict the class of an image
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the image file from the request
        image_file = request.files['imageInput']

        # Pre-process the image
        prep_image = preprocess_image(image_file)

        # Make predictions using the model
        prediction = model.predict(prep_image)

        # Get the predicted class and confidence
        predicted_class = class_names[prediction.argmax()]
        return render_template("index.html", predicted_class=predicted_class)
        

if __name__ == "__main__":
    app.run(debug=True)


