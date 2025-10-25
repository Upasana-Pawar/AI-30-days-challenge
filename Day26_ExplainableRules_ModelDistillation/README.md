# Day 26 — Explainable Decision Rules & Model Distillation

## Overview
Distill a RandomForest black-box into an interpretable Decision Tree surrogate. Evaluate fidelity (how well surrogate mimics the black-box) and surrogate accuracy (vs true labels), extract human-readable rules via `sklearn.tree.export_text`, and save tree visualization.

