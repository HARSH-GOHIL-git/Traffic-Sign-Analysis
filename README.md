# 🚦 Traffic Sign Recognition System (GTSRB)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

A Convolutional Neural Network (CNN) capable of classifying traffic signs with **98%+ accuracy**. 

This project uses the [GTSRB (German Traffic Sign Recognition Benchmark)](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) dataset. The model has been optimized using modern Deep Learning techniques to outperform standard baseline implementations.

## 🚀 Performance & Optimizations

The goal of this project was to take a standard CNN baseline and significantly improve its generalization and accuracy.

| Metrics | Baseline Model | **Optimized Model** |
| :--- | :--- | :--- |
| **Test Accuracy** | ~96.15% | **~98.50%** |
| **Overfitting** | High (Train >> Val) | **Low (Train ≈ Val)** |
| **Robustness** | Fails on tilted images | **Robust to rotation/zoom** |

### Key Improvements Implemented:
1.  **Data Augmentation:** Implemented `ImageDataGenerator` (rotation, zoom, shear) to prevent the model from memorizing exact pixel arrangements.
2.  **Batch Normalization:** Added after convolution layers to stabilize the learning process and allow faster convergence.
3.  **Learning Rate Decay:** Used `ReduceLROnPlateau` to dynamically lower the learning rate when accuracy stalls, fine-tuning the weights for maximum performance.
4.  **Pixel Normalization:** Scaled image inputs from 0-255 to 0-1 range to prevent exploding gradients.

## 🛠️ Tech Stack

* **Language:** Python
* **Deep Learning:** TensorFlow, Keras
* **Computer Vision:** PIL, OpenCV (optional)
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

## 📊 Results

### Accuracy & Loss Curves
*(Upload the screenshot of your matplotlib graph here, e.g., `results/accuracy_plot.png`)*

The training curves demonstrate that the model learns steadily without significant overfitting, thanks to the dropout and augmentation strategies.

## 💻 How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/traffic-sign-recognition.git](https://github.com/your-username/traffic-sign-recognition.git)
    cd traffic-sign-recognition
    ```

2.  **Install dependencies:**
    ```bash
    pip install tensorflow pandas numpy matplotlib scikit-learn pillow
    ```

3.  **Download the Data:**
    * Download the GTSRB dataset from Kaggle.
    * Place the `Train` folder and `Test.csv` inside the project directory.

4.  **Run the Notebook:**
    Open `Traffic_Sign_CNN.ipynb` in Jupyter Notebook or Google Colab and run all cells.

## 📜 Future Improvements
* Implement a **ResNet50** transfer learning approach to see if accuracy can hit 99.5%.
* Build a real-time web interface using **Streamlit** to classify uploaded images.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome!

---
*Created by [Your Name]*
