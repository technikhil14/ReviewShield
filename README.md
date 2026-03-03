# 🛡️ ReviewShield — Fake Product Review Detection

![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-green)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-REST-teal)

Detects fake Amazon product reviews using structured metadata and 
behavioral text pattern analysis — no NLP, pure ML.

##  Results
| Metric | Score |
|--------|-------|
| Accuracy | 82% |
| ROC-AUC | 0.90 |
| Dataset | 40K Amazon Reviews |
| Best Model | XGBoost |

## 🛠️ Tech Stack
- **ML:** XGBoost, Scikit-learn, SHAP
- **API:** FastAPI
- **Dashboard:** Streamlit
- **Container:** Docker

##  Run with Docker
```bash
docker pull nikhilbhadoriya/reviewshield
docker run -p 8000:8000 -p 8501:8501 nikhilbhadoriya/reviewshield
```

##  Key EDA Findings
- Dataset is perfectly balanced (50% fake, 50% genuine)
- Fake reviews have significantly lower lexical diversity
- Rating alone has 0.00 correlation with fake/genuine label
- `unique_word_ratio` is the strongest discriminating feature

##  API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/predict` | POST | Single review prediction |
| `/predict/bulk` | POST | Bulk CSV prediction |

##  Project Structure
```
ReviewShield/
├── Model/           ← trained model files
├── API/
│   ├── main.py      ← FastAPI app
│   ├── utils.py     ← feature engineering
│   ├── app.py       ← Streamlit dashboard
│   ├── Dockerfile
│   └── docker-compose.yml
└── Notebook/
    └── reviewshield_eda.ipynb
```

##  Local Setup
# Clone repo
```
git clone https://github.com/technikhil14/ReviewShield.git

# Install dependencies
pip install -r API/requirements.txt

# Run FastAPI
cd API
uvicorn main:app --reload

# Run Streamlit (new terminal)
streamlit run app.py
```
