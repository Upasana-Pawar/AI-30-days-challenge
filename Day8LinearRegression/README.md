# 🧮 Day 8 — Linear Regression: Predicting House Prices (30-Day AI Challenge)

**Challenge Focus:** Supervised Learning → Regression  
**Model Used:** Linear Regression (Baseline) + Ridge Regularization  
**Dataset:** California Housing Dataset (from `sklearn.datasets`)  
**Author:** Upasana  
**Date:** (Replace with today’s date)

---

## 🎯 What I Built

For **Day 8** of my 30-Day AI Challenge, I created a **Linear Regression model** that predicts house prices (median home value) using real data from the **California Housing dataset**.  
This project taught me how regression models estimate continuous outcomes and how to evaluate them using regression metrics.

---

## 🧩 Step-by-Step Workflow

1. **Loaded the dataset** using `fetch_california_housing()` from scikit-learn.  
2. **Performed basic EDA** (checked shape, head, correlations, summary statistics).  
3. **Split the data** into training and testing sets (80/20).  
4. **Scaled the features** using `StandardScaler`.  
5. **Trained a Linear Regression model** to predict median house values.  
6. **Evaluated model performance** using:
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)
   - Mean Absolute Error (MAE)
   - R² Score  
7. **Visualized results** with:
   - Predicted vs Actual scatter plot
   - Residuals distribution plot  
8. **Tried Ridge Regression** to reduce overfitting.  
9. **Saved the model & scaler** for reuse.

---

## 📊 Results

| Metric | Linear Regression | Ridge Regression |
|:--|:--:|:--:|
| **RMSE** | ~0.72 | ~0.71 |
| **MAE** | ~0.55 | ~0.54 |
| **R² Score** | ~0.61 | ~0.63 |

🧾 *Interpretation:*  
The Ridge model performed slightly better by penalizing large coefficients, reducing variance, and improving R². The results show a good baseline for regression tasks.

---

## 📈 Visualizations

- **Correlation Heatmap** – explores relationships between features.  
- **Predicted vs Actual Plot** – checks alignment between real and predicted values.  
- **Residual Distribution Plot** – examines errors (should be centered near zero).

All figures are saved inside the `figures/` folder:

