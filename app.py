from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input
from tensorflow.keras.datasets import mnist
# import cv2

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'mnist_model.h5'
ALLOWED_EXTENSIONS = {'png','jpg','jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# MODEL_PATH = os.path.join(MODEL_FOLDER, 'mnist_model.h5')


os.makedirs(UPLOAD_FOLDER,exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_existing_model():
    try:
        if os.path.exists(MODEL_PATH):
            print("Loading model...")
            return load_model(MODEL_PATH)
        else:
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def preprocess_image(image_path):

    img = tf.io.read_file(image_path)
    # Decode image
    img = tf.io.decode_image(img, channels=1)
    # Resize image
    img = tf.image.resize(img, [28, 28])
    # Convert to numpy array and normalize
    img_array = img.numpy()
    img_array = img_array.reshape(1, 28, 28, 1).astype('float32')
    img_array /= 255.0
    return img_array
    # return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'Success':False,'error':'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'Success':False,'error':'No file selected'})
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'],filename)
            file.save(filepath)

            processed_image = preprocess_image(filepath)
            prediction = model.predict(processed_image)
            predicted_digit = np.argmax(prediction[0])
            confidence = float(prediction[0][predicted_digit])

            os.remove(filepath)

            return jsonify({
                'success':True,
                'prediction': int(predicted_digit),
                'confidence':round(confidence*100,2)
            })
        except Exception as e:
            return jsonify({'success':False,'error':str(e)})
        
    return jsonify({'success':False,'error':'invalid file type'})

if __name__ == '__main__':
    model = load_existing_model()
    app.run(debug=True)