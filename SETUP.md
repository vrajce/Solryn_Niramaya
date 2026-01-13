# 🚀 Quick Setup Guide - Niramaya CDSS

This guide will help you set up the project on your local machine in minutes.

---

## 📋 Prerequisites

Before you begin, make sure you have:
- **Python 3.9 or higher** → [Download Python](https://www.python.org/downloads/)
- **Node.js 18 or higher** → [Download Node.js](https://nodejs.org/)
- **Git** → [Download Git](https://git-scm.com/downloads/)

Check your versions:
```bash
python --version  # or python3 --version
node --version
npm --version
git --version
```

---

## 🔥 Quick Start (Backend)

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd DAU_HACKATHON
```

### Step 2: Create Virtual Environment
```bash
# Create venv
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

**You'll know it's activated when you see `(venv)` in your terminal.**

### Step 3: Install Python Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**This will install:**
- FastAPI (API framework)
- TensorFlow (Deep learning)
- Ultralytics (YOLO models)
- OpenCV (Image processing)
- Pillow, NumPy, Streamlit, etc.

### Step 4: Get Model Files

⚠️ **Important:** Model files (`.pt`, `.h5`, `.keras`) are NOT in Git due to their large size.

**Ask your friend to share these files with you:**
- `router_model.h5`
- `bone_model.pt`
- `lung_model.pt`
- `lung-tb-model.pt`
- `best_model_auc.keras`
- `pneumonia_unet_v2.h5`

**Place them in the project root directory** (same folder as `api.py`).

### Step 5: Run the Backend API
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

**You should see:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
✅ Router model loaded
✅ Bone model loaded
✅ Lung/TB model loaded
...
```

**Test it:** Open browser → `http://localhost:8000/health`

---

## 🎨 Quick Start (Frontend)

### Step 1: Navigate to Frontend
```bash
cd frontend
```

### Step 2: Install Node Dependencies
```bash
npm install
```

**This will install:**
- React 19
- Vite (Build tool)
- React Router (Navigation)
- Lucide React (Icons)

### Step 3: Configure API Endpoint

Edit `frontend/src/services/api.js` and make sure the API URL matches your backend:

```javascript
const API_BASE_URL = 'http://localhost:8000';  // For local development
```

### Step 4: Run Frontend Dev Server
```bash
npm run dev
```

**You should see:**
```
  VITE v7.2.4  ready in 532 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

**Open browser** → `http://localhost:5173/`

---

## 🧪 Alternative: Run Streamlit Version

If you want to quickly test just the backend:

```bash
# Make sure venv is activated
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## 📦 Full Setup Commands (Copy-Paste)

### macOS/Linux:
```bash
# Backend
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# In a new terminal - Frontend
cd frontend
npm install
npm run dev
```

### Windows:
```bash
# Backend
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# In a new terminal - Frontend
cd frontend
npm install
npm run dev
```

---

## 🐛 Common Issues & Solutions

### Issue: `pip: command not found`
**Solution:** Try `pip3` instead of `pip` or reinstall Python with pip.

### Issue: `ModuleNotFoundError: No module named 'tensorflow'`
**Solution:** 
```bash
# Make sure venv is activated (you should see (venv) in terminal)
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Issue: `Model file not found: bone_model.pt`
**Solution:** Make sure all model files are in the root directory (same folder as `api.py`).

### Issue: `Port 8000 already in use`
**Solution:** 
```bash
# Find and kill the process
# macOS/Linux:
lsof -ti:8000 | xargs kill -9

# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Or use a different port:
uvicorn api:app --reload --port 8001
```

### Issue: Frontend can't connect to backend
**Solution:** 
1. Make sure backend is running on port 8000
2. Check CORS settings in `api.py` (should allow all origins for dev)
3. Verify API URL in `frontend/src/services/api.js`

### Issue: `npm install` fails
**Solution:** 
```bash
# Clear npm cache
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

---

## 🔄 Daily Development Workflow

```bash
# Day 1
git clone <repo>
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Get model files from friend
uvicorn api:app --reload

# Day 2 onwards
source venv/bin/activate  # Always activate venv first
uvicorn api:app --reload
```

---

## 📊 Verify Everything Works

1. **Backend Health Check:**
   ```bash
   curl http://localhost:8000/health
   # Should return: {"status": "healthy", "models_loaded": {...}}
   ```

2. **Frontend Loads:**
   - Open `http://localhost:5173/`
   - Should see the Niramaya landing page

3. **Upload Test:**
   - Click "Analyze Scan"
   - Upload a test X-ray
   - Should get results with AI analysis

---

## 💾 Sharing Model Files

Since model files are large (~500MB total), share them via:

**Option 1: Cloud Storage**
```bash
# Upload to Google Drive / Dropbox / OneDrive
# Share link with friend
```

**Option 2: USB Transfer**
```bash
# Copy all *.pt, *.h5, *.keras files to USB
```

**Option 3: Local Network (if on same WiFi)**
```bash
# On your machine (sender):
python -m http.server 8080

# Friend's machine (receiver):
wget http://<your-ip>:8080/bone_model.pt
wget http://<your-ip>:8080/router_model.h5
# etc for all model files
```

---

## 🎓 Learning Resources

- **FastAPI Docs:** https://fastapi.tiangolo.com/
- **React Docs:** https://react.dev/
- **YOLO Tutorial:** https://docs.ultralytics.com/
- **TensorFlow Guide:** https://www.tensorflow.org/tutorials

---

## 🆘 Still Stuck?

1. Check if Python/Node versions are correct
2. Make sure venv is activated (see `(venv)` in terminal)
3. Ensure all model files are present
4. Check terminal for specific error messages
5. Try running in a fresh terminal window

---

## 📝 Project Structure Reference

```
DAU_HACKATHON/
├── venv/                    # ← Your virtual environment (git ignored)
├── api.py                   # ← Main FastAPI backend
├── app.py                   # ← Streamlit alternative
├── requirements.txt         # ← Python dependencies
├── *.pt, *.h5, *.keras      # ← Model files (get from friend)
├── frontend/
│   ├── package.json         # ← Node dependencies
│   ├── src/
│   │   ├── App.jsx
│   │   └── components/
│   └── node_modules/        # ← Created by npm install
└── SETUP.md                 # ← This file!
```

---

**Happy Coding! 🚀**

If this guide helped, give it a ⭐ and share with others!
