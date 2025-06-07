# Project structure

```python
nlp/
├── .venv/                      # Unified virtual environment managed by uv
|
├── src/                        # All Python source code goes here
│   ├── backend/                # Python package for your API logic
│   │   ├── __init__.py         # Makes 'backend' a Python package
│   │   └── api.py              # Your main API server code (e.g., FastAPI, Flask)
│   │   └── services/           # (Optional) For business logic, DB interactions, etc.
│   │       ├── __init__.py
│   │       └── prediction_service.py
│   │   └── models/             # (Optional) For Pydantic models, DB models
│   │       ├── __init__.py
│   │       └── schemas.py
│   │
│   └── frontend/               # Python package for your Streamlit app logic
│       ├── __init__.py         # Makes 'frontend' a Python package
│       └── streamlit_app.py    # Your main Streamlit application file
│       └── components/         # (Optional) For reusable Streamlit components/functions
│           ├── __init__.py
│           └── sidebar.py
│
├── data/                       # For static data files, example datasets, etc.
│   ├── intermediary_data_test.tsv
│   ├── intermediary_data_train.tsv
│   ├── intermediary_data_val.tsv
│   ├── id2label.json
│   └── label2id.json
│
├── models/                     # For trained ML models, tokenizers, etc.
│   └── bert_news_classifier/
│       ├── config.json
│       ├── tf_model.h5
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── label2id.json       # A copy of your label mappings
│
├── pyproject.toml              # Central configuration for uv and project dependencies
├── uv.lock                     # Generated and managed by uv (do not edit manually)
├── .gitignore                  # Specifies files/folders Git should ignore
├── README.md                   # Project description
└── start_dev.sh                # (Optional) A simple script to start both
```
