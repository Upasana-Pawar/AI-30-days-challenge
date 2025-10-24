# Day 25 — Interpretable Probabilistic Models & Decision Thresholding

## Overview
This day focuses on turning calibrated probabilities into interpretable, operational decisions:
- Cost-aware threshold selection (minimize expected cost given FP/FN costs).
- Expected-value decision rule (act if expected utility > 0).
- Selective prediction / reject option: abstain on uncertain inputs and trade coverage vs accuracy.
- Simple human-readable rules derived from feature importances (median splits).

Uses scikit-learn breast_cancer dataset and a calibrated RandomForest (Platt scaling).

