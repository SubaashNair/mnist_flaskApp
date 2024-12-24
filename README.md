# MNIST Digit Recognition Flask App

A web application that uses a trained deep learning model to recognize handwritten digits from uploaded images. Built with Flask and TensorFlow.

## Features
- Upload image interface
- Real-time digit prediction
- Confidence score display
- Support for PNG, JPG, and JPEG formats

## Project Structure
```
mnist-flask-app/
├── app.py                  # Main Flask application
├── mnist_model.h5          # Pre-trained MNIST model
├── templates/
│   └── index.html         # Web interface template
├── uploads/               # Temporary storage for uploaded images
└── README.md
```

## Requirements
```
flask
tensorflow
pillow
numpy
werkzeug
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mnist-flask-app
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Ensure you have the pre-trained model file `mnist_model.h5` in the project root directory

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open a web browser and navigate to:
```
http://localhost:5000
```

## Usage

1. Access the web interface through your browser
2. Click the "Choose File" button to select an image
3. Select an image of a handwritten digit
4. The application will display:
   - The predicted digit
   - The confidence level of the prediction

## Technical Details

- Model: Convolutional Neural Network (CNN) trained on MNIST dataset
- Input: 28x28 grayscale images
- Output: Digit prediction (0-9) with confidence score
- Image preprocessing includes:
  - Resizing to 28x28
  - Conversion to grayscale
  - Normalization

## Limitations

- Only processes single-digit images
- Best results with clear, centered digits
- Image must be in PNG, JPG, or JPEG format
- Designed for digits written on white background with dark ink

## Contributing

Feel free to fork the project and submit pull requests with improvements.

## License

MIT License

## Contact

[Subashanan Nair/valiban12@gmail.com]

## Acknowledgments

- TensorFlow team for the MNIST dataset
- Flask framework developers

