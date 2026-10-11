# Skeleton Bone Age Assessment Using Advanced Image Preprocessing and CNN-Based Classification Techniques

An end-to-end academic research platform and full-stack machine learning system for pediatric bone age assessment using the **RSNA Pediatric Bone Age Challenge dataset**.

---

## 1. Project Overview

Pediatric bone age assessment (BAA) is a key clinical radiologic procedure for evaluating skeletal maturity, endocrine abnormalities, and growth disorders. This research platform investigates the empirical performance differences across:

- **4 Image Preprocessing Pipelines:**
  - `P0_baseline`: Standard resizing to 224×224, grayscale conversion, and min-max pixel normalization.
  - `P1_clahe`: Contrast Limited Adaptive Histogram Equalization (clip limit = 2.0, 8×8 grid) to enhance local bone trabeculae and epiphyseal contrast.
  - `P2_denoising`: Edge-preserving Gaussian filtering ($\sigma=0.8, k=3\times 3$) to attenuate detector noise.
  - `P3_sharpening`: Unsharp masking ($\alpha=1.5, \text{radius}=1.0$) to accentuate physis-metaphysis contours.

- **3 Pretrained CNN Architectures:**
  - **ResNet-18**: 18-layer residual network utilizing skip connections.
  - **DenseNet-121**: 121-layer dense architecture with direct feature reuse across layers.
  - **EfficientNet-B0**: Compound scaling architecture balancing depth, width, and resolution.

- **12-Cell Experimental Matrix:**
  Every combination is systematically trained, evaluated, and benchmarked under identical dataset splits and reproducible random seeds.

---

## 2. Classification Formulation

Continuous target bone ages (in months) are mapped to 4 academic discrete classes:

| Class ID | Chronological / Skeletal Range | Approximate Age | Cohort |
|:---:|:---:|:---:|:---:|
| **Class 0** | 0 – 48 months | 0 – 4.0 years | Infancy & Early Childhood |
| **Class 1** | 49 – 96 months | 4.1 – 8.0 years | Middle Childhood |
| **Class 2** | 97 – 144 months | 8.1 – 12.0 years | Late Childhood / Early Puberty |
| **Class 3** | 145 – 216 months | 12.1 – 18.0 years | Adolescence |

> **Note:** These initial ranges are project-defined categories for comparative CNN benchmarking and are not clinically validated diagnostic categories.

---

## 3. Project Architecture

```text
d:/MiniProject/
├── backend/
│   ├── app.py                     # Flask entry point with blueprint routing & CORS
│   ├── config.py                  # Central configuration (device, paths, hyperparameters)
│   ├── requirements.txt           # Backend dependencies (PyTorch, Torchvision, Flask, etc.)
│   ├── database/                  # SQLAlchemy SQLite database & ORM models
│   ├── routes/                    # API endpoints: health, dataset, preprocessing, models, experiments, predict
│   ├── preprocessing/             # Preprocessing modules (P0 - P3)
│   ├── models/                    # CNN architecture builders
│   ├── training/                  # Unified training loop with early stopping & checkpointing
│   └── evaluation/                # Confusion matrix & multi-class metric computation
│
├── frontend/                      # React 18+ SPA built with Vite and Tailwind CSS
│   ├── src/
│   │   ├── components/            # Navbar, Sidebar, StatCard, AcademicDisclaimer
│   │   ├── pages/                 # 8 Research Pages:
│   │   │                          # 1. Home / Overview
│   │   │                          # 2. Dataset Explorer
│   │   │                          # 3. Preprocessing Lab
│   │   │                          # 4. CNN Model Lab
│   │   │                          # 5. Experiment Matrix
│   │   │                          # 6. Comparative Analysis
│   │   │                          # 7. Prediction Demo
│   │   │                          # 8. Research Docs
│   │   └── services/              # API client module
│   ├── package.json
│   └── vite.config.js
│
├── data/
│   ├── raw/                       # Place RSNA dataset files here
│   │   └── README.md
│   ├── processed/                 # Cached preprocessed artifacts
│   └── metadata/                  # Train/validation/test split manifests
│
├── models/                        # Serialized PyTorch checkpoints (.pth)
├── experiments/                   # Experiment metadata
└── results/                       # Generated evaluation plots and confusion matrix heatmaps
```

---

## 4. Getting Started

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Node.js 18+ & npm
- PyTorch 2.x

### Backend Setup
1. Navigate to the project root and install Python dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```
2. Start the Flask backend server:
   ```bash
   python backend/app.py
   ```
   The backend will run on `http://127.0.0.1:5000`.

### Frontend Setup
1. Open a separate terminal, navigate to `frontend/`, and install dependencies:
   ```bash
   cd frontend
   npm install
   ```
2. Start the Vite development server:
   ```bash
   npm run dev
   ```
3. Open your browser at `http://localhost:5173`.

---

## 5. Dataset Ingestion (RSNA Pediatric Bone Age)

Place your dataset in `data/raw/`:
- `boneage-training-dataset.csv` (contains columns `id`, `boneage`, `male`)
- Hand radiographs (`.png` files) in `data/raw/` or `data/raw/images/`

Use the **Dataset Explorer** page to validate image paths, check missing rows, and view continuous-to-discrete age distributions.

---

## 6. Academic Disclaimer

This software is developed strictly for academic research, algorithm benchmarking, and educational evaluation. It is **NOT** a medical diagnostic device, is **NOT** clinically certified, and **MUST NOT** be used for clinical diagnosis or treatment planning.
