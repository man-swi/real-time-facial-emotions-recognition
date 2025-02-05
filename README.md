# RealTime Facial Emotions Recognition using CNN and OpenCV

Below is a live demonstration of the project in action. The interface shows real-time emotion detection using a webcam, where a CNN model (trained on the FER-2013 dataset) identifies facial expressions and annotates them on the video feed.

![Real-time Emotion Detection](https://github.com/user-attachments/assets/213b154e-1ef3-4bc8-ad67-783c6105f526)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Credits](#credits)
- [License](#license)

---

## Overview

Facial Emotion Recognition involves detecting and classifying the emotion expressed on a human face. This project uses a deep learning approach with a Convolutional Neural Network (CNN) to recognize facial expressions in real time. A webcam feeds live video into an OpenCV-powered interface, which detects faces, draws bounding boxes, and annotates the predicted emotions.

---

## Features

- **Real-time Emotion Detection:** Uses OpenCV to capture live video from a webcam.
- **CNN-Based Classification:** The emotion recognition model is built with TensorFlow/Keras and is trained on the FER-2013 dataset.
- **Multiple Emotions:** The model classifies facial expressions into seven categories:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Sad
  - Surprise
  - Neutral
- **Visualization:** Provides live annotations on the video feed, along with training history and performance metrics.

---

## Dataset

The project uses the FER-2013 dataset, which comprises 35,887 grayscale face images of size 48x48 pixels. Each image is labeled with one of seven emotion categories.

**Download the Dataset:**

You can download the original FER-2013 dataset from Kaggle:  
[Kaggle - Challenges in Representation Learning: Facial Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

![Dataset Sample](https://github.com/user-attachments/assets/cea915ec-fc61-4431-85a4-a5ada582fb69)
*(Ensure you accept the competition rules on Kaggle to access the dataset.)*

---

## Model Architecture

The CNN model architecture for this project includes:

- **Input Layer:** Accepts 48x48 grayscale images.
- **Convolutional Blocks:** Multiple convolutional layers with ELU activations, Batch Normalization, and MaxPooling layers to extract features.
- **Fully Connected Layers:** Dense layers with Dropout for regularization to mitigate overfitting.
- **Output Layer:** A Softmax layer that outputs probability scores for the seven emotion classes.

The model seamlessly integrates with OpenCV to perform real-time predictions on video frames.

---

## Installation

### Prerequisites

- Python 3.x
- Git

### Required Python Libraries

Install the necessary libraries using pip:

```sh
pip install tensorflow keras numpy pandas matplotlib opencv-python scikit-learn scikit-plot seaborn
```

*Optional:* If you are handling large files with Git, consider installing Git LFS as per the repository documentation.

---

## Usage

1. **Train the Model:**
   - Execute `train_model.py` to preprocess the data, train the CNN, and save the trained model.
   
   ```sh
   python train_model.py
   ```

2. **Real-time Emotion Detection:**
   - Run `real_time_emotion.py` to start the webcam interface for live emotion recognition.
   
   ```sh
   python real_time_emotion.py
   ```

*Note:* Ensure your webcam is connected and accessible by OpenCV.

---

## Credits

- **FER-2013 Dataset:** Provided by the Kaggle [Facial Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).
- **Libraries:** TensorFlow, Keras, OpenCV, and others.

---

## License

This project is licensed under the [MIT License](LICENSE).
---
