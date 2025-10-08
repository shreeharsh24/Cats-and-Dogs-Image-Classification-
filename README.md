# Cats-and-Dogs-Image-Classification-
Image classification of cats and dogs using deep learning and a CNN model in Python using Tensorflow,Keras


# 🐶🐱 Cats vs Dogs Image Classification using CNN

This project builds a **Convolutional Neural Network (CNN)** to classify images of **cats and dogs** using the **Dogs vs. Cats dataset** from Kaggle.
It demonstrates deep learning concepts such as image preprocessing, data augmentation, model training, and evaluation using **TensorFlow** and **Keras**.

---

## 📘 Project Overview

Image classification is one of the most common computer vision tasks. This project utilizes a CNN model to automatically distinguish between images of cats and dogs.

The dataset contains **25,000 labeled images** of cats and dogs (12,500 of each). The model learns from the training set and predicts whether an unseen image depicts a cat or a dog.

---

## 📂 Dataset

**Source:** [Dogs vs. Cats Dataset (Kaggle)](https://www.kaggle.com/datasets/salader/dogs-vs-cats)
This dataset was originally part of a Kaggle competition hosted by Microsoft Research.

**Dataset Structure:**

```
/train
   ├── cat.0.jpg
   ├── cat.1.jpg
   ├── dog.0.jpg
   └── dog.1.jpg
/test
   ├── cat.5001.jpg
   └── dog.5001.jpg
```

---

## 🧠 Model Architecture

The CNN is built using **TensorFlow Keras Sequential API**.

**Architecture Summary:**

1. **Conv2D + ReLU** layers for feature extraction
2. **MaxPooling2D** layers for spatial reduction
3. **BatchNormalization** and **Dropout** for regularization
4. **Flatten** layer to convert feature maps into vectors
5. **Dense** layers for classification
6. **Sigmoid output** for binary classification (cat/dog)

Example code snippet:

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(256,256,3)),
    MaxPooling2D(),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),
    Dropout(0.3),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

---

## ⚙️ Installation and Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/cats-vs-dogs-cnn.git
cd cats-vs-dogs-cnn
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Download the dataset

Create a `kaggle.json` file with your Kaggle API key and place it in the project root directory.

Then run:

```bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d salader/dogs-vs-cats
!unzip dogs-vs-cats.zip -d ./data
```

---

## 📊 Model Training

```python
history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=15
)
```

**Evaluation Metrics:**

* Training accuracy
* Validation accuracy
* Loss curves

You can visualize training progress with:

```python
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
```

---

## 📈 Results

| Metric   | Training | Validation |
| :------- | :------- | :--------- |
| Accuracy | ~97%     | ~94%       |
| Loss     | 0.08     | 0.12       |

Confusion matrix and accuracy plots show the model generalizes well to unseen data.

---

## 🧩 Technologies Used

* Python 3.8+
* TensorFlow / Keras
* NumPy / Pandas
* Matplotlib / Seaborn
* Kaggle API

---

## 💾 requirements.txt

```
tensorflow>=2.10.0
keras>=2.10.0
numpy>=1.24.0
pandas>=1.5.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
opencv-python>=4.8.0
Pillow>=9.5.0
jupyter>=1.0.0
```
---

## 📚 References

* [Kaggle: Dogs vs. Cats Dataset](https://www.kaggle.com/datasets/salader/dogs-vs-cats)
* Chollet, F. (2017). *Deep Learning with Python.*
* TensorFlow & Keras Official Documentation


**Shree Harsh**
🎓 Jain University


---

## 📝 License

This project is licensed under the **MIT License**.
See the [LICENSE](LICENSE) file for details.
