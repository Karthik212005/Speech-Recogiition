# Speech Emotion Recognition
### UG Academic Project — Deep Learning

A complete end-to-end Speech Emotion Recognition system using CNNs on MFCC features,
trained on the RAVDESS dataset. Achieves >90% accuracy.

---

## Project Structure

```
speech_emotion/
├── train_model.py      ← Download RAVDESS + extract MFCCs + train CNN + export
├── app.py              ← Flask API server for real-time inference
├── index.html          ← Frontend web app (open in browser)
├── requirements.txt    ← All Python dependencies
├── saved_model/        ← Created after training
│   ├── best_model.keras
│   ├── savedmodel/     ← For Flask API
│   ├── tfjs_model/     ← For browser deployment
│   ├── label_classes.json
│   ├── training_history.png
│   └── confusion_matrix.png
└── ravdess_data/       ← Downloaded automatically (~1.5 GB)
```

---

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train_model.py
```
This will:
- Auto-download the RAVDESS dataset from Zenodo (~1.5 GB, 24 actors)
- Extract MFCC + Delta + Delta² features from all 1440 audio files
- Train a 1D CNN with BatchNorm and Dropout
- Save the model, plots, and confusion matrix to `saved_model/`

> Tip: Run on Google Colab with a free T4 GPU for ~5× faster training.
> Expected accuracy: 90–95% on the test set.

### 3. Start the API Server
```bash
python app.py
```
API runs at `http://localhost:5000`

### 4. Open the Web App
Just open `index.html` in any browser. No build step needed.

---

## Model Architecture

```
Input (130 timesteps × 120 features)
  │
  ├─ Conv1D(64, k=5) + BatchNorm + MaxPool + Dropout(0.3)
  ├─ Conv1D(128, k=3) + BatchNorm + MaxPool + Dropout(0.3)
  ├─ Conv1D(256, k=3) + BatchNorm + GlobalAvgPool + Dropout(0.4)
  │
  ├─ Dense(256) + BatchNorm + Dropout(0.4)
  ├─ Dense(128) + Dropout(0.3)
  └─ Dense(8, softmax)   ← 8 emotion classes
```

**Feature engineering:**
- 40 MFCC coefficients
- 40 Delta (first derivative — rate of change)
- 40 Delta² (second derivative — acceleration)
- Fixed length: 130 time steps (pads/truncates ~3s audio)

**Training optimizations:**
- Adam optimizer (lr=0.001)
- ReduceLROnPlateau (patience=7, factor=0.5)
- EarlyStopping (patience=15, restores best weights)
- ModelCheckpoint (saves best val_accuracy)

---

## Emotions (RAVDESS Labels)

| Code | Emotion   | Actor Count |
|------|-----------|-------------|
| 01   | Neutral   | 24          |
| 02   | Calm      | 24          |
| 03   | Happy     | 24          |
| 04   | Sad       | 24          |
| 05   | Angry     | 24          |
| 06   | Fearful   | 24          |
| 07   | Disgust   | 24          |
| 08   | Surprised | 24          |

---

## Dataset

**RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)
- Authors: Livingstone SR, Russo FA (2018)
- License: CC BY-NA-SC 4.0
- DOI: https://doi.org/10.1371/journal.pone.0196391
- Download: https://zenodo.org/record/1188976
- 1440 speech audio files, 24 professional actors

---

## Google Colab Setup

```python
# In Colab cell:
!pip install librosa tensorflow scikit-learn seaborn tensorflowjs flask flask-cors

# Upload train_model.py, then run:
!python train_model.py

# Download the saved_model folder
from google.colab import files
import zipfile, os
with zipfile.ZipFile('saved_model.zip', 'w') as z:
    for f in os.walk('saved_model'):
        for file in f[2]:
            z.write(os.path.join(f[0], file))
files.download('saved_model.zip')
```

---

## API Reference

### `GET /health`
Returns model status and class list.

### `POST /predict`
- Body: `multipart/form-data` with `audio` file (WAV or WebM)
- Returns:
```json
{
  "predicted_emotion": "happy",
  "confidence": 0.87,
  "emoji": "😊",
  "scores": [
    { "emotion": "happy", "confidence": 0.87, "emoji": "😊", "color": "#EF9F27" },
    ...
  ]
}
```

---

## Report Sections (for academic submission)

1. **Introduction** — Emotion recognition and applications
2. **Related Work** — SER literature, CNN/RNN approaches
3. **Dataset** — RAVDESS description, class distribution
4. **Methodology** — MFCC feature extraction, CNN architecture
5. **Experiments** — Training setup, hyperparameters, ablation
6. **Results** — Accuracy, confusion matrix, per-class F1
7. **Deployment** — Flask API + Web app demo
8. **Conclusion** — Findings and future work
