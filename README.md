# 🧠 Day 6 — Classification: Diabetes Prediction (30-Day AI Challenge)

**Challenge Focus:** Supervised Learning → Binary Classification  
**Model Used:** Logistic Regression  
**Dataset:** Pima Indians Diabetes Dataset  
**Author:** Upasana  
**Date:** (Replace with today’s date)

---

## 🎯 What I Built

For **Day 6** of my 30-Day AI Challenge, I built a **machine learning model** that predicts whether a person has diabetes based on various medical parameters such as glucose level, blood pressure, BMI, and age.

The model uses **Logistic Regression** (a simple yet powerful classification algorithm) and provides performance metrics like **accuracy, precision, recall, and F1 score**.  
I also visualized the results using a **confusion matrix**.

---

## 🧩 What I Did Step-by-Step

1. Loaded the dataset from a public GitHub source using **pandas**.  
2. Explored basic statistics and column meanings.  
3. Split the data into **features (X)** and **target (y)**, where `Outcome = 1` indicates diabetes.  
4. Used **train_test_split** to divide the dataset (80% training / 20% testing).  
5. Standardized the data using **StandardScaler** to bring all numeric features to a similar scale.  
6. Trained a **Logistic Regression model** from **scikit-learn**.  
7. Predicted diabetes outcomes for the test set.  
8. Evaluated the model using accuracy, precision, recall, and F1 score.  
9. Plotted a **Confusion Matrix** using **matplotlib** to visualize correct and incorrect predictions.  

---

## 📊 Results

| Metric | Value |
|:--|:--:|
| **Accuracy** | 0.753 |
| **Precision** | 0.649 |
| **Recall** | 0.673 |
| **F1 Score** | 0.661 |

🧾 **Interpretation:**  
The model correctly classified about **75%** of all test samples.  
Precision and recall are reasonably balanced, giving a good baseline model.  
It sometimes predicts diabetes when it isn’t present, but overall captures most true diabetic cases.

---

## 🧠 What I Learnt

- The difference between **classification** and **regression** problems.  
- How **Logistic Regression** works — predicting probabilities and converting them into 0/1 labels.  
- The importance of **data scaling** before model training.  
- How to interpret **accuracy, precision, recall, and F1-score**.  
- How to analyze a **confusion matrix** to see where the model makes mistakes.  
- A deeper understanding of the **end-to-end ML workflow** — from loading data to evaluating results.  
- How to resolve **Unicode/emoji errors** on Windows terminals.

---

## 🧰 Tech Stack

- **Language:** Python 3.10+  
- **Libraries:** pandas, scikit-learn, matplotlib  
- **Environment:** VS Code / Anaconda  
- **OS:** Windows 11  

---

