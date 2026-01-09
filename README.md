# 🚦 Traffic Sign Analysis (GTSRB)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

A Convolutional Neural Network (CNN) capable of classifying traffic signs with **98%+ accuracy**.

This project uses the [GTSRB (German Traffic Sign Recognition Benchmark)](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) dataset. The model has been significantly optimized using modern Deep Learning techniques (Batch Normalization, Augmentation) to outperform standard baseline implementations found in basic tutorials.

## 🚀 Performance & Optimizations

The primary goal of this repository was to take a standard CNN baseline and improve its generalization and accuracy through architectural changes.

| Metrics | Baseline Model | **Optimized Model** |
| :--- | :--- | :--- |
| **Test Accuracy** | ~96.15% | **~98.50%** |
| **Overfitting** | High (Train >> Val) | **Low (Train ≈ Val)** |
| **Robustness** | Fails on tilted images | **Robust to rotation/zoom** |

### Key Improvements Implemented:
1.  **Data Augmentation:** Implemented `ImageDataGenerator` with rotation, zoom, and shear parameters to prevent the model from memorizing exact pixel arrangements.
2.  **Batch Normalization:** Added `BatchNormalization()` layers after convolutions to stabilize the learning process and allow for faster convergence.
3.  **Learning Rate Decay:** Used `ReduceLROnPlateau` to dynamically lower the learning rate when accuracy stalls, allowing the model to find the global minimum.
4.  **Pixel Normalization:** Scaled image inputs from `0-255` to `0-1` range to prevent exploding gradients.

## 🛠️ Tech Stack

* **Language:** Python
* **Deep Learning:** TensorFlow, Keras
* **Computer Vision:** PIL, OpenCV
* **Data Manipulation:** NumPy, Pandas, Scikit-learn
* **Visualization:** Matplotlib

## 📂 Dataset

The dataset consists of **43 Classes** of traffic signs (e.g., Stop, Speed Limit 20, Pedestrian Crossing).
* **Training Images:** ~39,000
* **Test Images:** ~12,000
* **Image Size:** Resized to `30x30` pixels

## 🧠 Model Architecture

The optimized architecture consists of:
1.  **Conv Block 1:** 2x Conv2D (32 filters) + BatchNormalization + MaxPool + Dropout
2.  **Conv Block 2:** 2x Conv2D (64 filters) + BatchNormalization + MaxPool + Dropout
3.  **Fully Connected:** Flatten -> Dense (256 units) -> Dropout -> Output (43 units, Softmax)

## 💻 How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/HARSH-GOHIL-git/Traffic-Sign-Analysis.git](https://github.com/HARSH-GOHIL-git/Traffic-Sign-Analysis.git)
    cd Traffic-Sign-Analysis
    ```

2.  **Install dependencies:**
    ```bash
    pip install tensorflow pandas numpy matplotlib scikit-learn pillow
    ```

3.  **Setup Data:**
    * Download the GTSRB dataset (Train folder and Test.csv).
    * Ensure the path in the script matches your local directory structure.

4.  **Run the Notebook:**
    Open the Jupyter Notebook and execute the cells to train the model and view the accuracy plots.

## 📜 Future Improvements
* Implement a **ResNet50** transfer learning approach to push accuracy towards 99.5%.
* Build a real-time web interface using **Streamlit** to classify uploaded images.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome!

---
*Maintained by [HARSH-GOHIL-git](https://github.com/HARSH-GOHIL-git)*
