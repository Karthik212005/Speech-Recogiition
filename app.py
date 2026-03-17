"""
Speech Emotion Recognition - Flask API
Serves the trained model for real-time inference via HTTP endpoints.

Run: python app.py
"""

import os
import json
import numpy as np
import librosa
import tempfile

from flask import Flask, request, jsonify
from flask_cors import CORS

import tensorflow as tf
from tensorflow import keras

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
MODEL_PATH  = './saved_model/savedmodel.keras'
LABELS_PATH = './saved_model/label_classes.json'
SAMPLE_RATE = 22050
DURATION    = 3
N_MFCC      = 40
MAX_LEN     = 130

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
#  LOAD MODEL AT STARTUP
# ─────────────────────────────────────────────
print("Loading model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"\n\nModel not found at: {MODEL_PATH}\n"
        f"Please run 'python train_model.py' first to train and save the model.\n"
    )

if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError(
        f"\n\nLabel file not found at: {LABELS_PATH}\n"
        f"Please run 'python train_model.py' first.\n"
    )

# Load using keras.models.load_model (Keras 3 compatible)
model = keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, 'r') as f:
    CLASSES = json.load(f)

EMOTION_META = {
    'neutral':   {'emoji': '😐', 'color': '#888780'},
    'calm':      {'emoji': '😌', 'color': '#1D9E75'},
    'happy':     {'emoji': '😊', 'color': '#EF9F27'},
    'sad':       {'emoji': '😢', 'color': '#378ADD'},
    'angry':     {'emoji': '😡', 'color': '#E24B4A'},
    'fearful':   {'emoji': '😨', 'color': '#7F77DD'},
    'disgust':   {'emoji': '🤢', 'color': '#D85A30'},
    'surprised': {'emoji': '😲', 'color': '#D4537E'},
}

print(f"Model loaded successfully!")
print(f"Classes: {CLASSES}")


# ─────────────────────────────────────────────
#  FEATURE EXTRACTION
# ─────────────────────────────────────────────
def extract_features(audio_bytes):
    """
    Accepts raw audio bytes, returns MFCC feature array shape (130, 120).
    Writes to a temp file so librosa can read it regardless of format.
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        audio, sr = librosa.load(tmp_path, sr=SAMPLE_RATE, duration=DURATION)
        audio     = librosa.util.normalize(audio)

        mfcc   = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        delta  = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        features = np.vstack([mfcc, delta, delta2]).T   # (time, 120)

        if features.shape[0] < MAX_LEN:
            pad      = MAX_LEN - features.shape[0]
            features = np.pad(features, ((0, pad), (0, 0)), mode='constant')
        else:
            features = features[:MAX_LEN, :]

        return features.astype(np.float32)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'classes': CLASSES})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts multipart/form-data with 'audio' file (WAV or WebM).
    Returns JSON with emotion predictions and confidence scores.
    """
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_bytes = request.files['audio'].read()

    try:
        # Extract features
        features = extract_features(audio_bytes)
        features = np.expand_dims(features, axis=0)   # (1, 130, 120)

        # Run prediction using keras model directly
        probs = model.predict(features, verbose=0)[0]   # shape (num_classes,)

        pred_idx   = int(np.argmax(probs))
        pred_label = CLASSES[pred_idx]
        confidence = float(probs[pred_idx])

        # Build scores for all classes
        scores = [
            {
                'emotion':    cls,
                'confidence': float(probs[i]),
                'emoji':      EMOTION_META.get(cls, {}).get('emoji', ''),
                'color':      EMOTION_META.get(cls, {}).get('color', '#888'),
            }
            for i, cls in enumerate(CLASSES)
        ]
        scores.sort(key=lambda x: x['confidence'], reverse=True)

        return jsonify({
            'predicted_emotion': pred_label,
            'confidence':        confidence,
            'scores':            scores,
            'emoji':             EMOTION_META.get(pred_label, {}).get('emoji', ''),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  Speech Emotion Recognition API")
    print("  Running on http://localhost:5000")
    print("  Open index.html in your browser")
    print("=" * 50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
