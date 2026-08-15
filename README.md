# 🦴 Bone Fracture AI Detection

An AI-powered bone fracture detection project that uses Machine Learning and OpenCV image processing to analyze X-ray images and classify them as **Fractured** or **Not Fractured**.

The project trains an SVM classifier using a labeled X-ray dataset and allows users to provide their own X-ray image for prediction.

## ✨ Features

- 🧠 Machine-learning-based fracture classification
- 🦴 Fractured / Not Fractured prediction
- 📊 Prediction confidence score
- 🖼️ Support for user-provided X-ray images
- 🔬 Grayscale image preprocessing
- 🌫️ Gaussian blur for noise reduction
- ⚙️ Morphological image processing
- 📐 Canny edge detection
- 🔎 Contour detection
- 💾 Automatic model saving
- 🖼️ X-ray visualization
- 📁 Separate training and testing datasets

## 🏗️ Architecture

X-Ray Dataset
      |
      +-------------------+
      |                   |
    TRAIN                TEST
      |                   |
      v                   v
Image Preprocessing   Model Evaluation
      |
      v
Feature Extraction
      |
      v
     SVM
      |
      v
Trained Model
      |
      v
User X-Ray Image
      |
      v
Preprocessing
      |
      v
AI Prediction
      |
      +----------------------+
      |                      |
      v                      v
 FRACTURED            NOT FRACTURED
      |                      |
      +----------+-----------+
                 |
                 v
          Confidence Score
                 |
                 v
       OpenCV Visualization

## 📂 Project Structure

Bone-Fracture/
│
├── app.py
├── fracture_model.pkl
├── fracture_result.jpg
├── README.md
│
└── Bone_Fracture_Binary_Classification/
    │
    ├── train/
    │   ├── fractured/
    │   └── not fractured/
    │
    └── test/
        ├── fractured/
        └── not fractured/

The dataset is not included in this repository.

Download the dataset separately and place it inside the project directory.

## 🧰 Technologies

Python
OpenCV
NumPy
Matplotlib
Scikit-learn
Support Vector Machine (SVM)

## 📊 Dataset

This project uses a Bone Fracture Binary Classification X-ray dataset from Kaggle.

The dataset contains two classes:

fractured
not fractured

Expected dataset structure:

Bone_Fracture_Binary_Classification/
│
├── train/
│   ├── fractured/
│   └── not fractured/
│
└── test/
    ├── fractured/
    └── not fractured/

## ⚙️ Installation

Make sure Python is installed.

Install the required packages:

pip install opencv-python numpy matplotlib scikit-learn

## 🚀 Usage

Run the application:

python app.py

The application will:

1. Load the training dataset.
2. Train the SVM classifier.
3. Load the test dataset.
4. Evaluate the model.
5. Display test accuracy.
6. Save the trained model.
7. Ask for a user X-ray image.
8. Predict Fractured or Not Fractured.
9. Display confidence.
10. Process and display the X-ray.
11. Save the processed image.

## 🔮 Example

When the program runs:

==========================================
          BONE FRACTURE AI
==========================================

Enter your X-ray image path: fracture.webp

X-ray loaded successfully!

==========================================
          BONE FRACTURE AI
==========================================

Prediction: FRACTURED

Confidence: 91.34 %

==========================================

The result can also be:

Prediction: NOT FRACTURED

Confidence: 88.21 %

The actual confidence depends on the trained model and input image.

## 🖼️ Image Processing

The X-ray is processed using the following pipeline:

X-Ray Image
     |
     v
Grayscale Conversion
     |
     v
Gaussian Blur
     |
     v
Erosion
     |
     v
Dilation
     |
     v
Canny Edge Detection
     |
     v
Contour Detection
     |
     v
Highlighted Image

The application displays:

1. Original X-ray
2. Canny Edge Detection
3. Highlighted Image

The processed image is saved as:

fracture_result.jpg

## 🤖 Machine Learning

The current implementation uses a Support Vector Machine (SVM).

The images are:

1. Converted to grayscale.
2. Resized to 64 × 64 pixels.
3. Normalized between 0 and 1.
4. Flattened into numerical features.
5. Passed to the SVM classifier.

The classifier uses two classes:

0 = NOT FRACTURED
1 = FRACTURED

The trained model is saved as:

fracture_model.pkl

## 📈 Model Evaluation

The model is evaluated using images from the test directory.

Example:

==========================================
             MODEL RESULT
==========================================

Test Accuracy: 87.50 %

==========================================

The actual accuracy is calculated when the application runs.

## 🛠️ Future Improvements

- Replace SVM with a CNN.
- Add transfer learning.
- Use ResNet or EfficientNet.
- Use a Vision Transformer.
- Add data augmentation.
- Add a Flask web interface.
- Add image upload functionality.
- Add confusion matrix.
- Add precision, recall and F1-score.
- Add fracture localization.
- Add image segmentation.
- Add Grad-CAM visualization.
- Improve model generalization.
- Add Docker support.

## ⚠️ Disclaimer

This project is intended for educational and research purposes only.

The model prediction is not a medical diagnosis and should not be used to make clinical decisions.

The OpenCV edge and contour visualization highlights image features but does not confirm the presence or location of a fracture.

A clinically usable system would require appropriate medical validation, expert evaluation, representative clinical data, and regulatory approval.

## 📜 License

This project is provided for educational and research purposes.

If you use a third-party dataset, follow the dataset's original license and usage requirements.
