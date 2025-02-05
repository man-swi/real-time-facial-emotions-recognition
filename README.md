# RealTime_Facial_Emotions_Recognition using CNN and OpenCV

This repository contains a computer vision project for real-time facial emotion recognition. The project uses a Convolutional Neural Network (CNN) trained on the FER-2013 dataset and integrates OpenCV for live emotion detection via a webcam.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Demonstration](#demonstration)
- [Credits](#credits)
- [License](#license)

---

## Overview

Facial Emotion Recognition involves detecting and classifying the emotion expressed on a human face. This project uses deep learning (CNN) to recognize facial expressions in real-time. A webcam feeds live video to an OpenCV-powered interface that draws bounding boxes around detected faces and annotates them with predicted emotions.

---

## Features

- *Real-time Emotion Detection:* Utilizes OpenCV to capture live video from a webcam.
- *CNN-Based Classification:* The emotion recognition model is built with TensorFlow/Keras and trained on the FER-2013 dataset.
- *Multiple Emotions:* The model classifies facial expressions into seven distinct categories:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Sad
  - Surprise
  - Neutral
- *Visualization:* Live annotation of emotions on video feed, plus training history and performance metrics.

---

## Dataset

The project uses the FER-2013 dataset which consists of 35,887 grayscale face images of size 48x48 pixels. Each image is labeled with one of seven emotion categories.

*Download the Dataset:*

You can download the original FER-2013 dataset from Kaggle:
[Kaggle - Challenges in Representation Learning: Facial Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

![Image](https://github.com/user-attachments/assets/12f45cc0-6fe2-41f5-affa-08c930b33775)

Note: Ensure you accept the competition rules on Kaggle to access the dataset.

---

## Model Architecture

The CNN model architecture used for this project includes:

- *Input Layer:* 48x48 grayscale images.
- *Convolutional Blocks:* Multiple convolutional layers with ELU activations, Batch Normalization, and MaxPooling layers to extract facial features.
- *Fully Connected Layers:* Dense layers with Dropout for regularization to reduce overfitting.
- *Output Layer:* A Softmax layer that outputs probabilities for seven emotion classes.

The model is integrated with OpenCV to perform real-time predictions on video frames captured from a webcam.

---

## Installation

### Prerequisites

- Python 3.x
- Git

### Required Python Libraries

Install the necessary libraries using pip:

sh
pip install tensorflow keras numpy pandas matplotlib opencv-python scikit-learn scikit-plot seaborn


Optional: For large file handling with Git, install Git LFS as described in the repository documentation.

## Usage

1. *Train the Model:*
   - Run train_model.py to preprocess the data, train the CNN, and save the trained model.
   
   sh
   python train_model.py
   

2. *Real-time Emotion Detection:*
   - Run real_time_emotion.py to start the webcam interface for live emotion recognition.
   
   sh
   python real_time_emotion.py
   

Note: Make sure your webcam is connected and accessible by OpenCV.

---

## Demonstration

Below is an example screenshot of the real-time emotion detection in action:

![Real-time Emotion Detection](![Image](https://github.com/user-attachments/assets/f8638926-adb6-4db3-ada8-f8a3d6ffb0af))

(Replace the above URL with the actual link to your demo image.)

---

## Credits

- *FER-2013 Dataset:* Provided by the Kaggle [Facial Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).
- *Libraries:* TensorFlow, Keras, OpenCV, and others.

---

## License

This project is licensed under the [MIT License](LICENSE).

---
