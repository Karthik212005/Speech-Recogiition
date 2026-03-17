"""
Speech Emotion Recognition - Training Script
Dataset: RAVDESS (downloaded automatically)
Model: 1D CNN on MFCC features
Target Accuracy: >90%

Windows Compatible — Keras 3 compatible (.keras save format)
Author: UG Academic Project
"""

import os
import zipfile
import urllib.request
import numpy as np
import librosa
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe on Windows
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
SAMPLE_RATE     = 22050
DURATION        = 3        # seconds per clip
N_MFCC          = 40       # MFCC coefficients
MAX_LEN         = 130      # fixed time steps
BATCH_SIZE      = 32
EPOCHS          = 100
DATASET_PATH    = "./ravdess_data"
MODEL_SAVE_PATH = "./saved_model"

# RAVDESS emotion mapping (position 3 in filename = emotion code)
EMOTION_MAP = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}


# ─────────────────────────────────────────────
#  STEP 1: DOWNLOAD RAVDESS DATASET
# ─────────────────────────────────────────────
def download_ravdess():
    """
    Downloads the RAVDESS audio speech dataset from Zenodo (open access).
    24 actors x 2 statements x 8 emotions = 1440 files total.
    Each actor's files are in a separate zip (~60 MB each).
    """
    os.makedirs(DATASET_PATH, exist_ok=True)
    base_url = "https://zenodo.org/record/1188976/files/"

    print("=" * 60)
    print("Downloading RAVDESS Dataset from Zenodo...")
    print("Total size ~1.5 GB. Please wait.")
    print("=" * 60)

    for actor_num in range(1, 25):
        actor_str = f"Actor_{actor_num:02d}"
        zip_name  = f"{actor_str}.zip"
        zip_path  = os.path.join(DATASET_PATH, zip_name)
        actor_dir = os.path.join(DATASET_PATH, actor_str)

        if os.path.exists(actor_dir):
            print(f"  [OK] {actor_str} already exists — skipping")
            continue

        url = base_url + zip_name
        print(f"  Downloading {actor_str}...", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(DATASET_PATH)
            os.remove(zip_path)
            print("done")
        except Exception as e:
            print(f"ERROR — {e}")

    print("\nAll actors downloaded.\n")


# ─────────────────────────────────────────────
#  STEP 2: FEATURE EXTRACTION
# ─────────────────────────────────────────────
def extract_mfcc(file_path):
    """
    Extract MFCC + Delta + Delta2 features from a WAV file.
    Returns array of shape (MAX_LEN, N_MFCC * 3) = (130, 120)

    Why these features?
    - MFCC   : captures spectral shape of the vocal tract
    - Delta  : first derivative — how spectrum changes over time
    - Delta2 : second derivative — acceleration of spectral change
    Together they give the model static + dynamic speech information.
    """
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        audio     = librosa.util.normalize(audio)

        mfcc   = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        delta  = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        features = np.vstack([mfcc, delta, delta2]).T   # (time, 120)

        # Pad or truncate to fixed MAX_LEN
        if features.shape[0] < MAX_LEN:
            pad      = MAX_LEN - features.shape[0]
            features = np.pad(features, ((0, pad), (0, 0)), mode='constant')
        else:
            features = features[:MAX_LEN, :]

        return features

    except Exception as e:
        print(f"  Warning: Skipping {os.path.basename(file_path)}: {e}")
        return None


def load_dataset():
    """
    Walk through all actor folders, extract features, build X and y arrays.
    RAVDESS filename format:
      modality-vocalChannel-emotion-intensity-statement-repetition-actor.wav
      e.g. 03-01-05-01-01-01-12.wav  ->  emotion code = '05' (angry)
    """
    print("Extracting MFCC features from audio files...")
    X, y  = [], []
    skipped = 0

    for actor_dir in sorted(os.listdir(DATASET_PATH)):
        actor_path = os.path.join(DATASET_PATH, actor_dir)
        if not os.path.isdir(actor_path):
            continue

        for fname in sorted(os.listdir(actor_path)):
            if not fname.endswith('.wav'):
                continue

            parts        = fname.replace('.wav', '').split('-')
            emotion_code = parts[2] if len(parts) >= 3 else None

            if emotion_code not in EMOTION_MAP:
                skipped += 1
                continue

            feats = extract_mfcc(os.path.join(actor_path, fname))
            if feats is not None:
                X.append(feats)
                y.append(EMOTION_MAP[emotion_code])
            else:
                skipped += 1

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    print(f"\n  Loaded  : {len(X)} samples")
    print(f"  Skipped : {skipped}")
    print(f"  Shape   : {X.shape}  (samples x time x features)")
    print("\n  Class distribution:")
    for cls in sorted(set(y)):
        print(f"    {cls:12s}: {np.sum(y == cls)}")

    return X, y


# ─────────────────────────────────────────────
#  STEP 3: MODEL ARCHITECTURE
# ─────────────────────────────────────────────
def build_model(input_shape, num_classes):
    """
    1D CNN for temporal sequence classification.

    Architecture:
      Input (130, 120)
        -> Conv Block 1: Conv1D(64)  x2 + BN + Pool + Dropout
        -> Conv Block 2: Conv1D(128) x2 + BN + Pool + Dropout
        -> Conv Block 3: Conv1D(256) x2 + BN + GlobalAvgPool + Dropout
        -> Dense(256) + BN + Dropout
        -> Dense(128) + Dropout
        -> Softmax(num_classes)
    """
    inp = keras.Input(shape=input_shape, name='mfcc_input')

    # Block 1
    x = layers.Conv1D(64, 5, padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, 5, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)

    # Block 2
    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)

    # Block 3
    x = layers.Conv1D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.4)(x)

    # Classifier head
    x   = layers.Dense(256, activation='relu')(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.4)(x)
    x   = layers.Dense(128, activation='relu')(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation='softmax', name='emotion')(x)

    return keras.Model(inp, out, name='SER_CNN')


# ─────────────────────────────────────────────
#  STEP 4: TRAIN
# ─────────────────────────────────────────────
def train():
    # 1. Download dataset
    download_ravdess()

    # 2. Extract features
    X, y_raw = load_dataset()

    # 3. Encode labels
    le          = LabelEncoder()
    y           = le.fit_transform(y_raw)
    num_classes = len(le.classes_)
    print(f"\nClasses ({num_classes}): {list(le.classes_)}")

    # Save label classes for Flask API inference
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    with open(os.path.join(MODEL_SAVE_PATH, 'label_classes.json'), 'w') as f:
        json.dump(list(le.classes_), f)
    print(f"Saved label_classes.json")

    # 4. Split: 70% train / 15% val / 15% test
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.176, stratify=y_tmp, random_state=42)

    print(f"\nSplit -> Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    # 5. One-hot encode targets
    y_tr = keras.utils.to_categorical(y_train, num_classes)
    y_vl = keras.utils.to_categorical(y_val,   num_classes)
    y_ts = keras.utils.to_categorical(y_test,  num_classes)

    # 6. Build + compile
    model = build_model(X_train.shape[1:], num_classes)
    model.summary()
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 7. Callbacks
    cb = [
        callbacks.EarlyStopping(
            monitor='val_accuracy', patience=15,
            restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=7, min_lr=1e-6, verbose=1),
        callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_SAVE_PATH, 'best_model.keras'),
            monitor='val_accuracy', save_best_only=True, verbose=1),
    ]

    # 8. Train
    print("\nStarting training...\n")
    history = model.fit(
        X_train, y_tr,
        validation_data=(X_val, y_vl),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cb,
        verbose=1
    )

    # 9. Evaluate on test set
    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    loss, acc = model.evaluate(X_test, y_ts, verbose=0)
    print(f"Accuracy : {acc*100:.2f}%")
    print(f"Loss     : {loss:.4f}")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\nPer-class Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # 10. Save plots
    _save_history_plot(history)
    _save_confusion_matrix(y_test, y_pred, le.classes_)

    # 11. Save model in .keras format (Keras 3 compatible — fixes save_format error)
    save_path = os.path.join(MODEL_SAVE_PATH, 'savedmodel.keras')
    model.save(save_path)
    print(f"\nModel saved -> {save_path}")
    print("\nTraining complete! Now run:  python app.py")

    return model, le, history


# ─────────────────────────────────────────────
#  PLOTTING HELPERS
# ─────────────────────────────────────────────
def _save_history_plot(history):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Training History', fontsize=14, fontweight='bold')

    axes[0].plot(history.history['accuracy'],     label='Train', lw=2)
    axes[0].plot(history.history['val_accuracy'], label='Val',   lw=2)
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history['loss'],     label='Train', lw=2)
    axes[1].plot(history.history['val_loss'], label='Val',   lw=2)
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(MODEL_SAVE_PATH, 'training_history.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved plot -> {out}")


def _save_confusion_matrix(y_true, y_pred, classes):
    cm     = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_pct, annot=True, fmt='.0%', cmap='Blues',
                xticklabels=classes, yticklabels=classes, linewidths=0.5)
    plt.title('Confusion Matrix (Normalised)', fontsize=14, fontweight='bold')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    out = os.path.join(MODEL_SAVE_PATH, 'confusion_matrix.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved plot -> {out}")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  SPEECH EMOTION RECOGNITION - TRAINING")
    print("  Model: 1D CNN  |  Dataset: RAVDESS")
    print("=" * 60 + "\n")

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"GPU detected: {gpus[0].name}\n")
    else:
        print("No GPU found - using CPU (expect 15-30 min training time)")
        print("Tip: Upload this file to Google Colab for free GPU training.\n")

    train()
