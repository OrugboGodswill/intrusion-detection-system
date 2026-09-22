# Sentinel IDS handoff

This project now runs as a Flask inference API and Streamlit dashboard. The starter Next.js files remain in the repository for reference, but Docker and local Python commands use the Python app.

## Add the trained artifacts

Create `models/` and place the serialized scikit-learn artifact at `models/random_forest_pipeline4.joblib`. If the pipeline expects a specific feature contract, add `models/metadata.json`:

```json
{"required_columns": ["feature_a", "feature_b"], "target_column": "label", "positive_label": 1}
```

The pipeline must own the same preprocessing used during training. The API never fabricates predictions: without the artifact it returns a clear `model_not_ready` response.

## Run locally

```bash
python -m pip install -r requirements.txt
python backend/app.py
streamlit run frontend/streamlit_app.py
```

Or run both services with `docker compose up --build`; open Streamlit at port 8501 and the API health endpoint at port 5000.

## CSV contract

Upload a headered CSV containing every `required_columns` entry. An optional target column is removed before inference. Results include `prediction` and, when supported by the pipeline, `confidence`.
