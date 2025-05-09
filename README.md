# Good vs Bad Image Classifier

A Streamlit web application that uses a Keras model to classify images as "Good" or "Bad". The application supports both image upload and webcam input.

## Features

- Image classification using a pre-trained Keras model
- Support for image upload
- Real-time webcam classification
- Confidence score display
- User-friendly interface

## Setup

1. Clone the repository:
```bash
git clone <your-repository-url>
cd <repository-name>
```

2. Create a virtual environment and activate it:
```bash
python -m venv .venv
.\.venv\Scripts\activate  # On Windows
source .venv/bin/activate  # On Unix/MacOS
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

## Requirements

- Python 3.10
- Streamlit
- TensorFlow/Keras
- OpenCV
- Pillow
- NumPy

## Usage

1. Choose your input method (Upload Image or Use Webcam)
2. If uploading an image, select an image file
3. If using webcam, allow camera access and click "Start Webcam"
4. View the classification results and confidence scores

## Model

The application uses a pre-trained Keras model (`keras_model.h5`) and corresponding labels (`labels.txt`). 