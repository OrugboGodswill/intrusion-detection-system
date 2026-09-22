# Sentinel IDS: AI-Powered Network Intrusion Detection System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1.1-green.svg)](https://flask.palletsprojects.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.44.1-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6.1-orange.svg)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Supported-cyan.svg)](https://www.docker.com/)

**Sentinel IDS** is an end-to-end Machine Learning and Cyber Security solution designed for automated batch network intrusion detection and explainable threat analysis. Powered by a pre-trained **Random Forest Pipeline** and **SHAP (SHapley Additive exPlanations)**, Sentinel IDS analyzes network traffic logs in CSV format, detects malicious behavior in real time, classifies specific cyber attack vectors, and provides security analysts with feature-level interpretability.

---

## 📌 Problem Statement & Solution

### The Challenge
Modern enterprise networks generate millions of traffic packets per minute. Traditional Signature-Based Intrusion Detection Systems (IDS) rely on known rules and fail to detect novel, zero-day, or anomalous attack vectors. Furthermore, many modern Deep Learning models act as "black boxes," leaving security operations center (SOC) teams without clear explanations for why an alert was triggered.

### The Sentinel Solution
- **Predictive Threat Classification**: ML model trained to detect complex multi-class attack categories (DoS, DDoS, Port Scans, Botnets, Brute Force, Web Attacks, and Benign traffic).
- **Explainable Security AI (XAI)**: Integrated SHAP engine that isolates exact network features (e.g., flow duration, packet length, flags) responsible for triggering security alerts.
- **Dual-Interface Architecture**:
  - **REST API (Flask)**: Scalable backend endpoint for batch predictions, CSV downloads, and explanation calculations.
  - **Interactive SOC Dashboard (Streamlit)**: Sleek, dark-mode user interface for network engineers and security analysts.
- **Containerized Orchestration**: Docker & Docker Compose setup for instant, production-ready deployment.

---

## 🚀 Key Features

- 🛡️ **Multi-Class Threat Classification**: Categorizes traffic into 7 distinct threat types + Benign traffic.
- 📊 **Batch CSV Processing**: Upload raw PCAP-derived network flow CSV files for instant analysis.
- 🔍 **SHAP Feature Importance & Attribution**: Global feature importance charts and per-row breakdown of contributing features.
- 📥 **Exportable Security Reports**: Direct CSV downloads of processed predictions complete with threat confidence scores.
- ⚡ **RESTful API Infrastructure**: Standardized JSON endpoints for easy integration with existing SIEM tools (Splunk, Elastic, Sentinel).
- 🐳 **Docker-Ready**: One-command deployment via Docker Compose for backend and frontend services.

---

## 🛠️ Tech Stack & Dependencies

### Data Science & Machine Learning
- **Python 3.10+**
- **Scikit-Learn**: Random Forest Pipeline with feature preprocessors.
- **SHAP (SHapley Additive exPlanations)**: Model explainability and feature contribution analysis.
- **Pandas & NumPy**: High-performance data structures and numerical processing.
- **Joblib**: Model serialization and artifact persistence.

### Web Backend & Frontend
- **Flask**: Microservices REST API for high-throughput inference endpoints.
- **Streamlit**: Interactive analytics UI and visualization dashboard.
- **Matplotlib**: Generation of SHAP summary plots in Base64 image format.
- **Next.js / React (TypeScript)**: Optional Next.js frontend scaffold for future web expansions.

---

## 🤖 Machine Learning Models & Class Mapping

The system relies on a serialized pipeline located at `models/random_forest_pipeline4.joblib` mapped via `models/metadata.json`.

### Class Mapping Table

| Class ID | Threat Category | Description |
| :---: | :--- | :--- |
| **0** | **Benign** | Normal, non-malicious network flow |
| **1** | **DoS** | Denial of Service attack |
| **2** | **DDoS** | Distributed Denial of Service attack |
| **3** | **PortScan** | Network reconnaissance / scanning activity |
| **4** | **Bot** | Botnet traffic / command-and-control communication |
| **7** | **Brute Force** | Password / credential brute-force attempts |
| **8** | **Web Attack** | Cross-Site Scripting (XSS), SQL Injection, etc. |

---

## 📁 Repository Structure

```
intrusion-detection-system/
├── backend/
│   ├── app.py                 # Flask REST API server (/api/health, /api/predict, /api/explain)
│   └── model_service.py       # ML Pipeline loader, validator, inference & SHAP explainability service
├── frontend/
│   └── streamlit_app.py       # Streamlit SOC Dashboard interface
├── models/
│   ├── random_forest_pipeline4.joblib # Serialized Scikit-Learn Random Forest Pipeline
│   ├── metadata.json          # Target columns, positive labels, and class mappings
│   └── .gitkeep
├── app/                       # Next.js frontend pages (App Router)
├── components/                # React / UI component library
├── lib/                       # Next.js utility functions
├── Dockerfile                 # Multi-stage container build definition
├── docker-compose.yml         # Container orchestration for Flask API & Streamlit UI
├── requirements.txt           # Python package dependencies
├── HANDOFF.md                 # System handoff & operational guidelines
├── unseen_test_data2.csv      # Sample test dataset for batch predictions
└── README.md                  # Project documentation
```

---

## 💻 Installation & Setup Guide

### Prerequisites
- **Python 3.10+** installed on your system
- **Git** installed
- *(Optional)* **Docker & Docker Compose**

### 1. Clone the Repository
```bash
git clone https://github.com/OrugboGodswill/intrusion-detection-system.git
cd intrusion-detection-system
```

### 2. Set Up Python Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Running the Application

### Option A: Running Locally (Recommended for Development)

1. **Start the Flask Backend API**:
   ```bash
   python backend/app.py
   ```
   *The REST API will start on `http://localhost:5000`.*

2. **In a new terminal window, start the Streamlit Dashboard**:
   ```bash
   streamlit run frontend/streamlit_app.py
   ```
   *The SOC Dashboard will launch automatically in your browser at `http://localhost:8501`.*

---

### Option B: Running with Docker Compose (Recommended for Production)

Run both the Flask backend API and Streamlit UI in isolated containers:

```bash
docker-compose up --build
```

- **Streamlit SOC Dashboard**: `http://localhost:8501`
- **Flask REST API Health Check**: `http://localhost:5000/api/health`

To stop the services:
```bash
docker-compose down
```

---

## 🔌 API Endpoints Reference

| Endpoint | Method | Description | Payload / Query |
| :--- | :---: | :--- | :--- |
| `/api/health` | `GET` | Health check endpoint and model artifact readiness | N/A |
| `/api/predict` | `POST` | Process network traffic CSV file and return JSON predictions | Form-data: `file=@traffic.csv` |
| `/api/predict/download` | `POST` | Process CSV and return annotated CSV file attachment | Form-data: `file=@traffic.csv` |
| `/api/explain` | `POST` | Generate SHAP feature importance plot (Base64) & row contributions | Form-data: `file=@traffic.csv` |

---

## 🧪 Testing with Sample Data

Sample datasets are included in the repository for quick validation:
1. Launch the Streamlit Dashboard (`http://localhost:8501`).
2. Drag and drop `unseen_test_data2.csv` into the **Upload traffic records** dropzone.
3. Click **Run batch detection** to view threat distribution and prediction probabilities.
4. Expand **Explainability and monitoring** and click **Generate SHAP explanation report** for deep feature interpretability.

---

## 📄 License & Attribution

Developed by **Orugbo Godswill** — Software Engineer & Data Scientist.  
Designed for network defense, cyber threat research, and automated security operation workflows.
