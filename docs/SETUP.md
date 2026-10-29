# Setup and prototype boundaries

[Main README](../README.md) · [Application views](../prediction/views.py)

## Dependencies

The [requirements file](../requirements.txt) lists packages imported by the current web application: Django, pandas, NumPy, requests, matplotlib and scikit-learn. The original environment lockfile is unavailable, so this is an inferred dependency list rather than a tested version lock.

The LSTM, GRU and MLP helper functions reference Keras names whose imports are commented out. Installing the listed dependencies does not enable a completed neural-network training workflow.

## Local workflow

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python manage.py check
python manage.py migrate
python manage.py runserver 127.0.0.1:8000
```

The settings file resolves the project root from its own location, so database and template paths do not depend on the original developer's home directory. Development settings retain `DEBUG=True`; the run command binds to the local machine.

## Current behavior and limitations

| Area | What the snapshot contains |
|---|---|
| Data retrieval | SNIRH request assembled from station and parameter values |
| Preprocessing | Date filtering, missing-value filling and plotting helpers |
| Data split | Chronological split and supervised-window preparation |
| State | Module-level dataset variables shared by requests |
| Date range | Fixed dates remain in preprocessing views |
| Request handling | Retrieval reads GET parameters; the form view also has a POST route |
| Model training | Helper functions exist, but Keras imports are disabled |
| Reproducibility | No original dependency lock or end-to-end test suite |

The page and preprocessing code require further integration work before all form paths can be described as functional. The sample climate CSVs are existing examples; they are not evidence of real river-forecasting accuracy.

## Source hygiene

`__pycache__/`, bytecode, virtual environments and newly generated local database/data outputs are ignored. The historical database snapshot and archives already in Git remain tracked. Python bytecode can be recreated by the interpreter and is not part of the application source.
