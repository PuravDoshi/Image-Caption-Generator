# 🧠 Image Caption Generator

This project implements an **Image Captioning System** that automatically generates natural-language descriptions for images. It combines **Convolutional Neural Networks (CNNs)** for visual feature extraction and **LSTMs** for sequence generation, trained on the **Flickr8k dataset**.

This project reflects my exploration of how vision and language models interact — a core part of my interest in **multimodal learning**. It complements my work on the **Pix2Pix GAN for low-light enhancement**, showing my ability to bridge perception and understanding in computer vision.

---

## 🚀 Features
- **End-to-end trainable model** that takes an image and produces captions.
- Uses **transfer learning (Xception)** for efficient feature extraction.
- **LSTM-based decoder** to learn linguistic structure and generate sentences.
- Modular and readable **Python implementation** for easy experimentation.
- **Tokenizer and sequence generator** designed for flexible dataset use.

---

## 🧩 Model Architecture
| Component | Description |
|------------|-------------|
| **CNN Encoder** | Extracts dense visual features using Xception pretrained on ImageNet. |
| **LSTM Decoder** | Generates captions word-by-word conditioned on image embeddings. |
| **Tokenizer** | Converts words into numerical sequences for training and inference. |
| **Greedy Search** | Produces the most probable caption during evaluation. |

---

## 📦 Requirements
```bash
pip install tensorflow keras numpy pandas pillow tqdm
```
## 📂 Project structure
```bash
ImageCaptionGenerator/
├── image_caption_generator.py   # Main script (train + test)
└── README.md                    # Project documentation
```
