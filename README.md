# Image Caption Generator 🖼️✍️

An end-to-end deep learning project that automatically generates descriptive captions for images. The application is built with a **CNN-LSTM** architecture and deployed using a **Flask** web interface.

---

## 🧠 Model Architecture

The model uses a classic encoder-decoder architecture, which is a popular and effective approach for image captioning.

### 1. **Encoder: InceptionV3**
* The encoder is a pre-trained **InceptionV3** model, a powerful Convolutional Neural Network (CNN).
* It processes an input image and extracts its most important features, converting the image into a compact feature vector (an embedding). This vector represents the "meaning" of the image.

### 2. **Decoder: LSTM**
* The decoder is a **Long Short-Term Memory (LSTM)** network, a type of Recurrent Neural Network (RNN).
* It takes the image feature vector from the encoder and learns to generate a sequence of words (a caption) that accurately describes the image. The model is trained to predict the next word in a sentence, given the image and all the previous words.

### 3. **Word Embeddings: Word2Vec**
* **Word2Vec** is used to create pre-trained word embeddings. This helps the model understand the relationships between words, leading to more coherent and contextually relevant captions.

---

## ⚙️ Tech Stack & Dependencies

* **Backend:** Python, Keras (with TensorFlow backend), Flask
* **Frontend:** HTML, CSS
* **Dataset:** Flickr8k (containing 8,000 images with 5 captions each)
* **Core Libraries:** `numpy`, `pandas`, `Pillow`, `tensorflow`, `keras`, `nltk`

---
