# QQQ ML Cybernetics Scaffolding

> End-to-end pipeline complete. Mock inference running, delayed feedback queue operational.  
> Awaiting Phase 0 data accumulation for live model training.

This repository contains the strict, low-latency, time-deterministic scaffolding for the QQQ Options Cybernetic Control Loop.

## Architecture Highlights
The pipeline collapses a 1-minute 0DTE/1DTE options snapshot (1,300 rows) into a single feature vector, generates an XGBoost inference, and routes the signal through an adaptive Cybernetic PID Controller.

The Cybernetic Controller is designed to natively handle **Delayed Feedback**.  
A prediction at time $t$ targets the return at $t+N$. The environment queues the prediction, waits $N$ minutes, retrieves the realized return, and retroactively updates the feature weights $w_i(t)$ using a PID rule distributed by SHAP attributions.

## Project Structure
```text
src/
├── features/
│   ├── validator.py     # Reject corrupt snapshots (e.g., missing greeks)
│   ├── buffer.py        # O(1) in-memory N-snapshot caching for time-derivatives
│   └── builder.py       # Deterministic collapse of 1300 rows -> 1 feature row
├── models/
│   ├── xgb_model.py     # Base classifier pipeline
│   └── explainer.py     # SHAP execution for local feature attributions
├── control/
│   ├── feedback.py      # Delay Queue + TTL Market Close logic
│   ├── pid.py           # Core P-I-D math for error scaling
│   ├── regime.py        # Classifier mapping state to {open, mid, close, high_vol}
│   └── loop.py          # Central decay rule applicator
├── training/
│   └── labeler.py       # Strictly isolated offline t+N logarithmic label generation
├── utils/
│   └── logger.py        # JSON outputting (stdout/file) for DB/Execution ingestion
└── pipeline.py          # E2E Event orchestration
```

## Running Tests
Ensure dependencies (`pandas, numpy, xgboost, shap, pytest`) are installed.
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
pytest tests/
```

See `guide.md` for instructions on integrating real data when Phase 0 concludes.
