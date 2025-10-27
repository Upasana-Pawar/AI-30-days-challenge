# Day 28 — Selective Rule Ensembles & Coverage Maximization

## Overview
Greedy selection of rules extracted from a surrogate decision tree to build a compact rule ensemble. The algorithm:
- extracts rules from a decision tree surrogate,
- computes per-rule coverage & precision,
- greedily selects rules that (a) increase uncovered coverage and (b) meet a minimum precision threshold,
- evaluates final rule-set coverage, accuracy vs ground truth, and fidelity vs black-box.

