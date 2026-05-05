# ASL Alphabet Recognition Web App
This project is a prototype for recognizing static American Sign Language (ASL) alphabet gestures using image classification.
The goal is to build a simple AI-powered hand gesture recognition system that can classify ASL letter images and eventually be extended into a web-based and real-time webcam demo.
目前第一版 prototype 先聚焦在 5 個 ASL 字母：
- A
- B
- C
- L
- Y
This smaller MVP version helps us verify the full pipeline first, including dataset preparation, model training, evaluation, image prediction, and future web integration.
---
## Project Motivation
Hand gestures are an important form of communication. For deaf or hard-of-hearing communities, sign language is a key communication tool. However, not everyone understands sign language, which may create communication barriers.
This project explores how machine learning and deep learning can be used to recognize static ASL hand gestures from images. Although this prototype does not perform full sign language translation, it provides a foundation for static hand gesture recognition and future real-time applications.
---
## Project Scope
This project is designed to match the course topics, including:
- AI / ML / DL concepts
- Python development with VS Code
- Scikit-learn machine learning baseline
- TensorFlow / Keras neural network model
- Image classification
- CNN-based hand gesture recognition
- Future Flask Web App integration
- Future real-time webcam gesture detection
---
## Dataset
This project uses the **ASL Alphabet Dataset**, which contains RGB hand gesture images organized by class folders.
The original dataset contains classes such as:
```text
A, B, C, ..., Z, del, nothing, space

For the first MVP version, we only use five classes:

A, B, C, L, Y

These five classes were selected because their hand shapes are visually different and easier to distinguish in an early prototype. This helps us validate the complete system pipeline before expanding to more ASL letters.
```
---

## Current Project Structure

sign-language-mnist-web-app/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── asl_alphabet_train/
│   │   ├── A/
│   │   ├── B/
│   │   ├── C/
│   │   ├── D/
│   │   ├── E/
│   │   ├── ...
│   │   ├── L/
│   │   ├── Y/
│   │   ├── Z/
│   │   ├── del/
│   │   ├── nothing/
│   │   └── space/
│   │
│   ├── asl_alphabet_train_mvp/
│   │   ├── A/
│   │   ├── B/
│   │   ├── C/
│   │   ├── L/
│   │   └── Y/
│   │
│   └── asl_alphabet_test/
│       ├── A_test.jpg
│       ├── B_test.jpg
│       ├── C_test.jpg
│       ├── L_test.jpg
│       ├── Y_test.jpg
│       └── ...
│
├── outputs/
│   ├── figures/
│   │   ├── cnn_mvp_sample_images.png
│   │   ├── cnn_mvp_training_accuracy.png
│   │   ├── cnn_mvp_training_loss.png
│   │   ├── cnn_mvp_confusion_matrix.png
│   │   ├── cnn_mvp_per_class_accuracy.png
│   │   ├── cnn_mvp_prediction_confidence_distribution.png
│   │   └── cnn_mvp_single_prediction.png
│   │
│   └── models/
│       ├── cnn_mvp_model.keras
│       ├── cnn_mvp_label_mapping.pkl
│       └── cnn_mvp_label_mapping.json
│
├── static/
│   └── uploads/
│
├── templates/
│   └── index.html
│
├── train_sklearn_baseline.py
├── train_tensorflow_cnn_mvp.py
└── predict_asl_cnn_mvp.py

---

## Main Files

train_sklearn_baseline.py

This file trains a traditional machine learning baseline model using Scikit-learn.

The images are processed as:

Image
↓
Grayscale
↓
Resize
↓
Flatten into pixel features
↓
Scikit-learn classifier

This baseline is used to compare traditional machine learning with deep learning.

---

### train_tensorflow_cnn_mvp.py

This is the main training file for the current MVP model.

It trains a small CNN model using TensorFlow / Keras on five ASL classes:

A, B, C, L, Y

The model pipeline is:

ASL image
↓
Resize image
↓
Rescale pixel values
↓
Conv2D layers extract image features
↓
MaxPooling reduces feature size
↓
Dense layers classify the gesture
↓
Output predicted ASL letter

The model outputs are saved to:

outputs/models/

The training graphs and evaluation results are saved to:

outputs/figures/



### predict_asl_cnn_mvp.py

This file loads the trained CNN MVP model and predicts a single ASL image.

The test image path can be changed inside the file:

TEST_LETTER = "C"
IMAGE_PATH = Path(f"data/asl_alphabet_test/{TEST_LETTER}_test.jpg")

Example:

TEST_LETTER = "A"

This will test:

data/asl_alphabet_test/A_test.jpg

The prediction result includes:

* Original image
* Resized image
* Top prediction probabilities
* Predicted class
* Confidence score



## Model Design

The current TensorFlow model is a small CNN model trained from scratch.

It does not use MobileNetV2 or transfer learning in the current MVP version. The first version uses a lightweight CNN architecture to reduce training time and make the implementation easier to explain.

Current CNN model components:

* Rescaling
* Conv2D
* MaxPooling2D
* Flatten
* Dense
* Dropout
* Softmax output



Why We Use a 5-Class MVP First

The complete ASL Alphabet dataset contains many classes, and some gestures are visually similar. Training all classes at once takes more time and may make early debugging harder.

Therefore, the current prototype first uses:

A, B, C, L, Y

These classes were selected because they are visually distinct and suitable for a first-stage demo.

This allows us to verify:

* Data loading
* Image preprocessing
* CNN model training
* Model evaluation
* Prediction pipeline
* Future Web App integration

After the MVP works correctly, the model can be expanded to more ASL letters.


## Installation

1. Create a virtual environment

python3 -m venv .venv
source .venv/bin/activate

2. Install dependencies

python -m pip install -r requirements.txt



## Requirements

The requirements.txt file should include:

tensorflow
numpy
pandas
matplotlib
scikit-learn
joblib
opencv-python
flask
pillow
ipykernel
jupyter



## How to Prepare the MVP Dataset

Create a smaller MVP dataset using five classes:

rm -rf data/asl_alphabet_train_mvp
mkdir -p data/asl_alphabet_train_mvp
cp -r data/asl_alphabet_train/A data/asl_alphabet_train_mvp/
cp -r data/asl_alphabet_train/B data/asl_alphabet_train_mvp/
cp -r data/asl_alphabet_train/C data/asl_alphabet_train_mvp/
cp -r data/asl_alphabet_train/L data/asl_alphabet_train_mvp/
cp -r data/asl_alphabet_train/Y data/asl_alphabet_train_mvp/

Check the folder:

ls data/asl_alphabet_train_mvp

Expected output:

A B C L Y



## How to Train the CNN MVP Model

Run:

python train_tensorflow_cnn_mvp.py

This will generate:

outputs/models/cnn_mvp_model.keras
outputs/models/cnn_mvp_label_mapping.pkl
outputs/models/cnn_mvp_label_mapping.json

And figures such as:

outputs/figures/cnn_mvp_sample_images.png
outputs/figures/cnn_mvp_training_accuracy.png
outputs/figures/cnn_mvp_training_loss.png
outputs/figures/cnn_mvp_confusion_matrix.png
outputs/figures/cnn_mvp_per_class_accuracy.png
outputs/figures/cnn_mvp_prediction_confidence_distribution.png


## How to Predict a Test Image

Open predict_asl_cnn_mvp.py and change:

TEST_LETTER = "C"

For example:

TEST_LETTER = "L"

Then run:

python predict_asl_cnn_mvp.py

The result figure will be saved as:

outputs/figures/cnn_mvp_single_prediction.png



## Current Output Examples

The model produces several useful charts for reporting:

Figure	Purpose
cnn_mvp_sample_images.png	Shows dataset image examples
cnn_mvp_training_accuracy.png	Shows training and validation accuracy
cnn_mvp_training_loss.png	Shows training and validation loss
cnn_mvp_confusion_matrix.png	Shows model classification performance
cnn_mvp_per_class_accuracy.png	Shows accuracy for each class
cnn_mvp_prediction_confidence_distribution.png	Shows prediction confidence distribution
cnn_mvp_single_prediction.png	Shows single image prediction result



## Web App Plan

The next stage is to build a Flask Web App.

Planned workflow:

User uploads an ASL image
↓
Flask backend receives the image
↓
Image is resized to the model input size
↓
CNN model predicts the ASL letter
↓
Web page displays prediction and confidence score

Planned files:

app.py
templates/index.html
static/uploads/



## Future Real-Time Detection Plan

The future real-time version can use webcam input.

Possible workflow:

Webcam frame
↓
Select center ROI area
↓
Resize image
↓
CNN model prediction
↓
Display predicted ASL letter in real time

This version may require additional preprocessing to handle:

* Different backgrounds
* Hand position changes
* Lighting differences
* User hand size and angle differences


Current Limitations

The current model is an MVP version and only supports:

A, B, C, L, Y

It may not work well on letters outside these classes.

Also, if the uploaded image is very different from the training dataset style, the model may produce incorrect predictions. For real-world use, more diverse training data and stronger preprocessing would be needed.



Future Improvements

Possible future improvements include:

* Expanding from 5 classes to more ASL alphabet classes
* Adding space, del, and nothing
* Improving the CNN architecture
* Using transfer learning such as MobileNetV2
* Adding real-time webcam prediction
* Adding hand detection or ROI cropping
* Improving model performance on real-world images
* Deploying the web app online



Summary

This project demonstrates a complete AI image classification workflow:

Dataset preparation
↓
Model training
↓
Model evaluation
↓
Single image prediction
↓
Future web app integration

The current MVP uses a TensorFlow CNN model to classify five static ASL alphabet gestures: A, B, C, L, and Y.