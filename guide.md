# Cybernetics Phase 0 to Phase 1 Guide

This guide details exactly how to transition the scaffolding into a live modeling environment once your Phase 0 data collection (15-20 days of 1-minute 0DTE/1DTE snapshots) is complete.

## Step 1: Data Integration (`src/features/builder.py`)

Currently, `builder.py` contains scaffolding mathematics. Your first step will be expanding the feature aggregations based on your EDA findings.

1. **Add Custom Features:** In `FeatureBuilder.build_features()`, expand the dictionary with features like IV Skew, Charm Exposure, or specific strike magnets.
2. **Buffer Utilization:** If your EDA reveals value in 5-minute aggregations (e.g. Gamma Acceleration), adjust the `WindowBuffer` capacity in `pipeline.py` and extract the oldest snapshot in `builder.py` to calculate the $\Delta$ derivative.
3. **Run Stability Tests:** Rerun `test_builder.py`. If a feature isn't deterministic (e.g., relying on a Pandas `.sample()` or index assumption without sorting), the test will fail. Fix it before proceeding.

## Step 2: Model Training (`src/training/labeler.py` & `src/models/xgb_model.py`)

You need a baseline capable of generating meaningful SHAP values before the cybernetic loop can take control.

1. **Generate Offline Targets:** Build a training script that iterates through your historical snapshots and calls `TargetLabeler.compute_labels()`. Ensure `target_minutes` aligns with your system (e.g., 5 or 15 minutes). Note: `Labeler` produces `1`, `0`, and `-1`. XGBoost binary classifier needs `1` and `0` — you may filter out the `0` (flat) dead-band or switch to `XGBRFClassifier` (multi-class).
2. **Train & Save:** Train XGBoost on the first 70% of days. DO NOT shuffle. Save the weights to a `.json` or `.ubj` file.
3. **Load in Pipeline:** Update `pipeline.py` to instantiate the model with the path: `self.model = XGBoostBaseline(model_path="weights.json")`.
4. **Activate SHAP:** In `src/models/explainer.py`, remove the `is_mock = True` flag and instantiate the real `shap.TreeExplainer(self.model.model)`.

## Step 3: Run the Cybernetic Loop

1. **Simulate a Stream:** Write an ingestion script that reads your raw parquet files row-by-row and constructs real `snapshot_df` inputs. Send them continuously to `PipelineOrchestrator.process_snapshot()`.
2. **Monitor the Logs:** The pipeline will start dumping JSON to `signals.log`. 
3. **Analyze cybernetic weight drift:** Plot the `feature_weights` over time from your log file. If a weight quickly drifts to zero, it means that feature is consistently providing incorrect signals relative to the XGB baseline. 

## Step 4: Parameter Tuning (`src/control/loop.py`)

Once you verify the system runs End-to-End without throwing errors on real data, you will need to tune the PID controller.
1. Increase `alpha` (e.g., to 0.1) if the model is too slow to react to an intraday regime shift.
2. Decrease `alpha` if the feature weights are oscillating wildly between 1-minute snapshots.
3. Once stable, introduce Integral (`ki`) and Derivative (`kd`) terms to the PID logic.
