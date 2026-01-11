# 🏥 Niramaya - AI-Powered Clinical Decision Support System

<p align="center">
  <img src="https://img.shields.io/badge/React-19.2-61DAFB?style=for-the-badge&logo=react&logoColor=white" alt="React"/>
  <img src="https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/YOLO-Ultralytics-00FFFF?style=for-the-badge&logo=yolo&logoColor=black" alt="YOLO"/>
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
</p>

<p align="center">
  <strong>Empowering resource-limited healthcare settings with automated, high-precision diagnostics</strong>
</p>

---

## 🌟 What is Niramaya?

**Niramaya** (Sanskrit for "free from disease") is an intelligent Clinical Decision Support System (CDSS) that brings the power of explainable AI to medical diagnostics. Built with love and purpose, it helps healthcare professionals detect **Bone Fractures**, **Tuberculosis (TB)**, and **Pneumonia** from X-ray scans with remarkable accuracy.

In many parts of the world, access to specialist radiologists is limited. Niramaya bridges this gap by providing instant, reliable analysis that works even with low-resolution scans and noisy data—because quality healthcare shouldn't depend on where you live.

---

## ✨ Key Features

### 🔬 **Multi-Disease Detection**
- **Bone Fractures** - Precise localization using a 2-phase detection algorithm (Standard + Deep Zoom)
- **Tuberculosis (TB)** - Lung field analysis with zone-specific findings
- **Pneumonia** - Classification with U-Net segmentation for affected area visualization

### 🎯 **Smart Routing**
Our intelligent router model automatically determines whether an X-ray is a bone scan, chest scan, or an invalid image—no manual selection required.

### 🧠 **AI + Clinical Fusion Engine**
We don't just rely on AI. Niramaya combines:
- **Visual AI Analysis** - Deep learning-based pattern recognition
- **Clinical Context** - Symptom-based risk scoring
- **Fusion Logic** - Smart combination for final diagnosis confidence

### 🗺️ **Contour Mapping & Explainability**
See exactly *where* and *why* the AI made its decision. Our contour overlays highlight affected regions, making the diagnosis transparent and trustworthy.

### 📊 **Risk Stratification**
Clear, actionable risk levels:
- 🔴 **HIGH RISK** - Immediate attention required
- 🟠 **MODERATE RISK** - Further review recommended
- 🟡 **CLINICAL WARNING** - Scan negative but symptoms concerning
- 🟢 **LOW RISK** - No significant findings

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (React)                        │
│   Landing Page → Upload Area → Results View → History        │
└─────────────────────┬───────────────────────────────────────┘
                      │ REST API
┌─────────────────────▼───────────────────────────────────────┐
│                    Backend (FastAPI)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Router    │→ │  Bone Model │  │    Lung Models      │  │
│  │  (TF/h5)    │  │   (YOLO)    │  │  TB + Pneumonia     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                          ↓                                   │
│              Clinical Symptom Fusion Engine                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.9+**
- **Node.js 18+** (for frontend)
- **Git**

### Backend Setup

```bash
# Clone the repository
git clone https://github.com/your-username/niramaya.git
cd niramaya

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the API server
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The app will be available at `http://localhost:5173`

---

## 📁 Project Structure

```
niramaya/
├── api.py                    # FastAPI backend with all endpoints
├── app.py                    # Streamlit version (alternative UI)
├── symptom_solver.py         # Clinical symptom analysis logic
├── requirements.txt          # Python dependencies
│
├── 🤖 Models
│   ├── router_model.h5       # Scan type classifier (Bone/Chest/Invalid)
│   ├── bone_model.pt         # YOLO model for fracture detection
│   ├── lung_model.pt         # YOLO model for TB detection
│   ├── best_model_auc.keras  # Pneumonia classifier
│   └── pneumonia_unet_v2.h5  # U-Net for pneumonia segmentation
│
├── frontend/                 # React application
│   ├── src/
│   │   ├── components/       # UI components
│   │   │   ├── LandingPage.jsx
│   │   │   ├── UploadArea.jsx
│   │   │   ├── ResultsView.jsx
│   │   │   └── HistoryView.jsx
│   │   └── services/
│   │       └── api.js        # API communication layer
│   └── package.json
│
├── 🐳 Docker
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── nginx/nginx.conf
│
└── 📜 Documentation
    ├── README.md
    └── DEPLOYMENT.md
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/analyze` | Analyze X-ray with symptoms |
| `POST` | `/analyze-chest` | Specialized chest analysis (TB/Pneumonia) |
| `GET` | `/history` | Retrieve scan history |
| `DELETE` | `/history/{scan_id}` | Delete specific scan |
| `DELETE` | `/history` | Clear all history |
| `GET` | `/health` | Check API and model status |

---

## 🐳 Docker Deployment

```bash
# Development (with hot reload)
docker-compose -f docker-compose.local.yml up --build

# Production
docker-compose -f docker-compose.prod.yml up -d
```

For detailed Azure VM deployment instructions, see [DEPLOYMENT.md](./DEPLOYMENT.md).

---

## 💡 How It Works

### 1. **Image Upload**
User uploads an X-ray scan through the intuitive drag-and-drop interface.

### 2. **Smart Routing**
The router model classifies the scan type with high confidence:
- Is it a bone X-ray?
- Is it a chest X-ray?
- Or is it an invalid/non-medical image?

### 3. **Specialist Analysis**
Based on routing:
- **Bone scans** → YOLO fracture detection with 2-phase analysis
- **Chest scans** → User selects TB or Pneumonia pathway
  - TB: YOLO-based detection with lung zone mapping
  - Pneumonia: Classification + U-Net segmentation

### 4. **Clinical Context**
The system considers patient symptoms:
- Pain, fever, swelling, deformity (for bone)
- Cough, night sweats, weight loss, blood in sputum (for TB/Pneumonia)

### 5. **Fusion & Results**
AI confidence + Clinical boost = Final risk assessment with explainable reasoning.

---

## 🎨 Screenshots

| Landing Page | Upload Interface | Results View |
|:------------:|:----------------:|:------------:|
| Clean, modern hero section | Drag & drop with symptom checklist | Detailed analysis with contour overlay |

---

## 🛡️ Disclaimer

> **Niramaya is designed as a decision-support tool for healthcare professionals, not a replacement for clinical judgment.** All results should be reviewed and validated by qualified medical personnel before making treatment decisions.

---

## 🤝 Contributing

We welcome contributions! Whether it's:
- 🐛 Bug fixes
- ✨ New features
- 📖 Documentation improvements
- 🧪 Additional model training

Feel free to open issues and pull requests.

---

## 📜 License

This project is developed for the **DAU Hackathon 2025**. Please contact the team for licensing inquiries.

---

## 💖 Acknowledgments

- **Ultralytics** for the amazing YOLO framework
- **TensorFlow** team for making deep learning accessible
- **React** community for the brilliant frontend ecosystem
- All the open-source medical imaging datasets that made this possible

---

## 👥 Meet the Team

Built with passion by students from **Dhirubhai Ambani University** 🎓

| Name | Role | Degree | Semester |
|:-----|:-----|:-------|:---------|
| **Vraj Parmar** | AI/ML Engineer | B. Tech | 6th |
| **Nemin Haria** | AI/ML Engineer | B. Tech | 6th |
| **Harsh Shah** | AI/ML Engineer | B. Tech | 6th |
| **Bhavya Sonigra** | Fullstack Engineer | B. Tech | 6th |
| **Jay Dolar** | Fullstack Engineer | B. Tech | 6th |

---

<p align="center">
  <strong>🩺 Because early detection saves lives 🩺</strong>
</p>

<p align="center">
  Made with ❤️ for the DAU Hackathon 2025
</p>
