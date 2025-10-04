# Day 5 — Linear Regression (Machine Learning Basics)

## 🎯 Goal
Learn the basics of **Machine Learning** using **Linear Regression** to predict house prices.  
Started with a simple one-feature model (median income) and then extended to multiple features.

---

## 📊 Dataset
- Used **California Housing Dataset** from scikit-learn.
- Rows: ~20,640, Columns: 9.
- Features include median income, house age, average rooms, population, etc.
- Target: `MedHouseVal` (median house value).

---

## ✅ What I Did
1. Loaded and explored the dataset (`fetch_california_housing`).
2. Built a **single-feature regression model** using only `MedInc`.
3. Evaluated the model:
   - **MSE:** ~0.709
   - **R²:** ~0.459 (explains ~46% of variance).
4. Built a **multi-feature regression model** using:
   - `MedInc`, `HouseAge`, `AveRooms`, `Population`, `AveOccup`
5. Compared performance:
   - Multi-feature model gave **lower MSE** and **higher R²** (better fit).
6. Plotted results:
   - `figures/day5_linear_regression.png` — Single feature (Income vs House Value).
   - `figures/day5_multi_feature_actual_vs_pred.png` — Actual vs Predicted (multi-feature).

---

## 📈 Results
- **Single Feature Model** (MedInc → MedHouseVal)  
  - MSE: ~0.709  
  - R²: ~0.459  

- **Multi Feature Model** (MedInc + HouseAge + AveRooms + Population + AveOccup)  
  - MSE: _(lower, exact value printed in script output)_  
  - R²: _(higher, better fit than 0.459)_

---

## 🔑 Learnings
- Linear regression can draw a "best fit line" to predict continuous values.  
- A single feature gives a rough estimate, but combining features improves accuracy.  
- **R²** tells us how much of the variation in the target is explained by the model.  
- Visualizing actual vs predicted values makes it easier to see model performance.  

---

## 📂 Files
- `day5_linear_regression.py` → Single feature regression  
- `day5_linear_regression_multifeature.py` → Multi-feature regression  
- `README.md` → This documentation  
- `figures/day5_linear_regression.png` → Scatterplot (income vs house price)  
- `figures/day5_multi_feature_actual_vs_pred.png` → Actual vs predicted plot  

---

