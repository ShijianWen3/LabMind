# LabMind

> 智能光谱数据分析与分类系统 / Intelligent Spectral Data Analysis & Classification System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/PCA-SVM-Algorithm-orange" alt="PCA+SVM">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

---

## Overview

LabMind is a web-based spectral data analysis application that combines **FastAPI** backend with **PCA+SVM** machine learning algorithms to provide intelligent classification and visualization for spectroscopic data.

### Key Features

- **📊 Data Visualization**: Interactive CSV file upload and spectral curve display
- **🔬 Two-Stage Classification**: Hierarchical PCA+SVM model
  - Stage 1: Qualified vs Defective
  - Stage 2: Defect type细分 (Class A/B/C)
- **🌐 Bilingual Support**: Chinese and English UI
- **💡 Smart Suggestions**: Context-specific experimental recommendations based on classification results

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI + Uvicorn |
| Frontend | HTML5 + Jinja2 + Chart.js + HTMX |
| Algorithm | PCA (Principal Component Analysis) + SVM (Support Vector Machine) |
| Deployment | Single executable (PyInstaller) |

---

## Project Structure

```
labmind/
├── main.py                 # FastAPI application & routes
├── infer.py                # Inference module
├── interface/
│   └── pca-svm.py          # PCA+SVM training & prediction
├── templates/
│   └── index.html          # Frontend UI
├── static/                 # Static assets
├── model/                  # Trained models (generated)
├── temp/                   # Temporary files (generated)
└── requirements.txt        # Dependencies
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python main.py
```

The application will open automatically in your browser at `http://127.0.0.1:8000`

### 3. Prepare Training Data

```
dataset/train/
├── class_A/   # 缺陷类型A - 漫反射问题
├── class_B/  # 缺陷类型B - 金刚石未压紧
├── class_C/  # 缺陷类型C - 背景测量问题
└── class_D/  # 合格样本
    └── *.csv  # Each CSV: x,y columns, ~8000 rows
```

### 4. Train the Model

```bash
python interface/pca-svm.py
```

---

## Usage

1. **Upload CSV**: Drag & drop or click to select spectral data files (.csv)
2. **Visualize**: View spectral curves in the chart area
3. **Classify**: Click "获取建议" / "Get Advice" to classify
4. **Result**: View classification result with confidence rate and experimental suggestions

---

## Classification Logic

```
输入光谱数据
    │
    ▼
┌─────────────────┐
│ Stage 1: PCA+SVM│ → Qualified (class_D) vs Defective
└────────┬────────┘
         │
         ▼ (if Defective)
┌─────────────────┐
│ Stage 2: PCA+SVM│ → Class A / Class B / Class C
└─────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│          Experimental Suggestion        │
├─────────────────────────────────────────┤
│ Class A → 建议使用漫反射                 │
│ Class B → 建议压紧金刚石                 │
│ Class C → 建议重新测量背景               │
│ Class D → 结果似乎不错                   │
└─────────────────────────────────────────┘
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Main page (zh/en) |
| POST | `/upload` | Upload CSV for visualization |
| POST | `/api/classify` | Classification inference |
| GET | `/api/translations` | Get translations |

---

## Screenshots

> Modern dark/light interface with JetBrains Mono font

| Feature | Description |
|---------|-------------|
| 文件上传 | Drag & drop CSV with visual feedback |
| 光谱显示 | Interactive Chart.js visualization |
| 分类结果 | Stage 1 + Stage 2 results with probability bars |
| 建议显示 | Context-aware experimental suggestions |

---

## License

MIT License

---

## Author

Developed with ❤️ for spectral analysis