# Day 23 — Uncertainty Estimation & Bayesian Methods

## Overview
This day demonstrates practical uncertainty estimation approaches:

1. **Deep Ensembles (classification)**: Train multiple models (RandomForest) with different seeds, average predictive probabilities and compute std as an epistemic uncertainty proxy.
2. **Bayesian Regression**: Use `BayesianRidge` on the diabetes dataset to obtain predictive mean and std (`predict(..., return_std=True)`).
3. **Optional MC Dropout**: Small Keras/TensorFlow demo for predictive uncertainty using dropout at inference time (runs only if TF is installed).

