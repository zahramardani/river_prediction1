# River Data Web Prototype — Django

A historical Django prototype for selecting hydrological stations, retrieving SNIRH data, displaying tables, filling missing values and inspecting time-series preparation.

**Author:** Zahra Mardani Korani  
[All projects](https://github.com/zahramardani/ML2PP-Artifacts/blob/main/PROJECTS.md) · [Setup and current limitations](docs/SETUP.md)

## Which river repository is this?

| Repository | Purpose |
|---|---|
| **This repository: `river_prediction1`** | Historical Django station-data and preprocessing interface |
| [river_prediction](https://github.com/zahramardani/river_prediction) | Separate synthetic forecasting, FastAPI, SQL and retrieval portfolio |
| [ML2PP-Artifacts](https://github.com/zahramardani/ML2PP-Artifacts) | ML2++ research artifacts and reference workflows |

This snapshot is an earlier prototype. It should not be treated as the complete current AI4RIVER platform or as a verified operational forecasting service.

## Repository map

| Path | Purpose |
|---|---|
| [manage.py](manage.py) | Django command entry point |
| [river_prediction/settings.py](river_prediction/settings.py) | Application configuration |
| [river_prediction/urls.py](river_prediction/urls.py) | Routes for data retrieval and preprocessing |
| [prediction/views.py](prediction/views.py) | SNIRH retrieval, table rendering, plots and window preparation |
| [prediction/models.py](prediction/models.py) | Deployment metadata model |
| [prediction/templates/](prediction/templates/) | Web interface templates |
| [prediction/migrations/](prediction/migrations/) | Database schema migration |
| [prediction/dataset/](prediction/dataset/) | Existing example climate CSVs |
| [river_prediction2.zip](river_prediction2.zip) | Historical archive |
| [requirements.txt](requirements.txt) | Dependencies inferred from active Python imports |

## Local inspection

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python manage.py check
python manage.py migrate
python manage.py runserver 127.0.0.1:8000
```

Use `.venv\Scripts\activate` on Windows. These are the local setup steps, not a claim that every workflow has been validated. Read [SETUP.md](docs/SETUP.md) for the prototype's known limitations.

Python bytecode caches are excluded from source control. The original database, CSVs and ZIP archive remain available at their existing paths.
