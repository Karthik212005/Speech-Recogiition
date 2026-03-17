# 🚀 Deploy Speech Emotion Recognition to GitHub Pages

This guide deploys your SER web app so it's accessible from any browser via a public URL like:
`https://YOUR-USERNAME.github.io/speech-emotion-recognition/`

No server required. The frontend runs entirely in the browser.

---

## Prerequisites

- A [GitHub account](https://github.com/signup) (free)
- [Git](https://git-scm.com/downloads) installed on your computer
- Your project files (index.html, train_model.py, app.py, etc.)

---

## Step 1 — Create a GitHub Repository

1. Go to [github.com/new](https://github.com/new)
2. Fill in:
   - **Repository name:** `speech-emotion-recognition`
   - **Description:** `Speech Emotion Recognition using CNN on RAVDESS dataset`
   - **Visibility:** Public ✅ (required for free GitHub Pages)
3. ✅ Check **"Add a README file"**
4. Click **"Create repository"**

---

## Step 2 — Upload Your Files

### Option A — Via GitHub Website (easiest)

1. Open your new repository on GitHub
2. Click **"Add file" → "Upload files"**
3. Drag and drop these files:
   - `index.html` ← **most important**
   - `train_model.py`
   - `app.py`
   - `requirements.txt`
   - `README.md`
4. Scroll down → type commit message: `"Initial commit: SER web app"`
5. Click **"Commit changes"**

### Option B — Via Git CLI

```bash
# Navigate to your project folder
cd /path/to/your/project

# Initialize git
git init

# Connect to your GitHub repo (replace YOUR-USERNAME)
git remote add origin https://github.com/YOUR-USERNAME/speech-emotion-recognition.git

# Stage all files
git add .

# Commit
git commit -m "Initial commit: SER web app"

# Push to GitHub
git branch -M main
git push -u origin main
```

---

## Step 3 — Enable GitHub Pages

1. In your repository, click the **"Settings"** tab (⚙️)
2. In the left sidebar, click **"Pages"**
3. Under **"Source"**, select:
   - Branch: **`main`**
   - Folder: **`/ (root)`**
4. Click **"Save"**
5. Wait 1–2 minutes for deployment

---

## Step 4 — Access Your Live Site

After deployment, your app is live at:

```
https://YOUR-USERNAME.github.io/speech-emotion-recognition/
```

GitHub will show a blue banner in the Pages settings with the exact URL.

> 💡 Bookmark this URL and share it with your professor or evaluators!

---

## Step 5 — Update the App (after changes)

Whenever you update `index.html` or other files:

```bash
git add .
git commit -m "Update: improved UI"
git push
```

GitHub Pages auto-redeploys in ~60 seconds.

---

## What Works on GitHub Pages (No Backend)

Since GitHub Pages is static hosting (no Python/Flask), the web app works like this:

| Feature | Status | Notes |
|---------|--------|-------|
| 🎙 Microphone recording | ✅ Works | Uses Web Audio API |
| 📂 File upload | ✅ Works | Any audio format |
| 🎵 RAVDESS samples | ✅ Works | Streamed from Zenodo |
| ⬇️ Download samples | ✅ Works | Direct Zenodo links |
| 🤖 AI Analysis (Flask API) | ⚠️ Demo mode | Shows simulated results |
| 📊 Confidence bars | ✅ Works | Animated visualization |
| 📱 Mobile responsive | ✅ Works | Full mobile support |

> When the Flask API (`app.py`) is running locally, full real predictions work. On GitHub Pages, the app gracefully falls back to simulated results with a toast notification.

---

## Optional — Full Deployment with Real Predictions

To get real ML predictions on the web, deploy the Flask backend to a cloud service:

### Option 1: Render.com (free tier)
```bash
# Add a requirements.txt (already have it)
# Add a Procfile:
echo "web: python app.py" > Procfile

# Add to git and push
git add Procfile
git commit -m "Add Procfile for Render"
git push
```
Then on render.com: New → Web Service → connect your GitHub repo.
Set start command: `python app.py`
Copy the URL (e.g., `https://ser-api.onrender.com`) and update `API_BASE` in `index.html`:
```js
const API_BASE = 'https://ser-api.onrender.com';
```

### Option 2: Railway.app (free $5/month credit)
Same process — connect GitHub repo, set `python app.py` as start command.

### Option 3: Google Colab + ngrok (for demo sessions)
```python
# In Colab:
!pip install flask flask-cors pyngrok
from pyngrok import ngrok
public_url = ngrok.connect(5000)
print("API URL:", public_url)
# Run app.py, update API_BASE in index.html with the ngrok URL
```

---

## Project Structure for GitHub

```
speech-emotion-recognition/
├── index.html          ← Web app (GitHub Pages serves this)
├── train_model.py      ← Model training script
├── app.py              ← Flask API backend
├── requirements.txt    ← Python dependencies
└── README.md           ← Project documentation
```

---

## README.md for GitHub (copy this)

```markdown
# Speech Emotion Recognition

A deep learning project that classifies emotions in speech audio using a 1D CNN trained on the RAVDESS dataset.

## Live Demo
👉 [https://YOUR-USERNAME.github.io/speech-emotion-recognition/](https://YOUR-USERNAME.github.io/speech-emotion-recognition/)

## Features
- 🎙 Real-time microphone recording with waveform visualization
- 📂 Upload any audio file (WAV, MP3, OGG, FLAC, M4A)
- 🎵 RAVDESS sample audio clips (downloadable)
- 📊 Confidence scores for all 8 emotion classes
- 🤖 1D CNN model trained on MFCC + Delta features

## Emotions
Neutral · Calm · Happy · Sad · Angry · Fearful · Disgust · Surprised

## Tech Stack
- Python · TensorFlow · librosa · Flask
- HTML/CSS/JavaScript · Web Audio API

## Dataset
[RAVDESS](https://zenodo.org/record/1188976) — Livingstone & Russo (2018)

## Run Locally
\`\`\`bash
pip install -r requirements.txt
python train_model.py   # Train model (~20 min)
python app.py           # Start API server
# Open index.html in browser
\`\`\`
```

---

## Troubleshooting

**"Page not found" after enabling Pages:**
Wait 2–3 minutes and hard-refresh the browser (Ctrl+Shift+R)

**"Permission denied" on git push:**
Use GitHub Desktop app or set up a Personal Access Token at github.com/settings/tokens

**Microphone not working on the live site:**
GitHub Pages uses HTTPS, which is required for microphone access. This should work fine.

**RAVDESS samples not playing:**
Zenodo may have CORS restrictions. The download links always work. For playback, the samples analyze fine when downloaded and uploaded via the file upload tab.
