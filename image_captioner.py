import os
import numpy as np
import pickle
from tqdm import tqdm
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

def load_captions(filepath):
    captions = {}
    with open(filepath, "r") as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            img, caption = parts
            img = img.split("#")[0]  # remove caption number
            caption = caption.lower().replace("-", " ")
            caption = " ".join([w for w in caption.split() if w.isalpha()])
            caption = f"<start> {caption} <end>"
            captions.setdefault(img, []).append(caption)
    return captions

def extract_features(image_dir):
    model = Xception(include_top=False, pooling="avg")
    features = {}
    for img_name in tqdm(os.listdir(image_dir), desc="Extracting features"):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(image_dir, img_name)
        image = Image.open(path).resize((299, 299))
        image = preprocess_input(np.expand_dims(np.array(image), axis=0))
        features[img_name] = model.predict(image, verbose=0)
    return features

def create_tokenizer(descriptions):
    lines = [caption for caps in descriptions.values() for caption in caps]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def max_caption_length(descriptions):
    lines = [caption for caps in descriptions.values() for caption in caps]
    return max(len(c.split()) for c in lines)

def create_sequences(tokenizer, max_len, captions, features, vocab_size):
    while True:
        for img, caps in captions.items():
            feature = features[img][0]
            for caption in caps:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_len)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    yield ([feature, in_seq], out_seq)


def build_model(vocab_size, max_len):
    # Image feature branch
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.4)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    # Text sequence branch
    inputs2 = Input(shape=(max_len,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = LSTM(256)(se1)

    # Merge branches
    decoder = add([fe2, se2])
    decoder = Dense(256, activation='relu')(decoder)
    outputs = Dense(vocab_size, activation='softmax')(decoder)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary()
    return model

if __name__ == "__main__":
    IMAGE_DIR = "/Users/puravdoshi/Downloads/Image Captioning Project/Flicker8k_Dataset"
    CAPTION_FILE = "/Users/puravdoshi/Downloads/Image Captioning Project/Flickr8k_text/Flickr8k.token.txt"

    # Step 1: Prepare data
    captions = load_captions(CAPTION_FILE)
    features = extract_features(IMAGE_DIR)
    tokenizer = create_tokenizer(captions)
    vocab_size = len(tokenizer.word_index) + 1
    max_len = max_caption_length(captions)

    print(f"Vocabulary Size: {vocab_size}, Max Length: {max_len}")
    pickle.dump(tokenizer, open("tokenizer.pkl", "wb"))
    pickle.dump(features, open("features.pkl", "wb"))

    # Step 2: Build model
    model = build_model(vocab_size, max_len)

    # Step 3: Train
    generator = create_sequences(tokenizer, max_len, captions, features, vocab_size)
    steps = sum(len(c.split()) for caps in captions.values() for c in caps)
    model.fit(generator, epochs=3, steps_per_epoch=steps // 32, verbose=1)

    # Step 4: Save
    os.makedirs("models", exist_ok=True)
    model.save("models/image_captioner.h5")
