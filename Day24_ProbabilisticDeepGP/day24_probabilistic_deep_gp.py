"""
Day24: Probabilistic Deep Learning & Gaussian Processes

Demonstrations:
- GaussianProcessRegressor (sklearn) on the diabetes dataset (1D feature view) -> mean +/- uncertainty bands.
- GaussianProcessClassifier (sklearn) on make_moons toy data -> probabilistic decision surface (predict_proba).
- Probabilistic Deep Learning:
    - Try TensorFlow Probability (TFP) Bayesian NN if both tensorflow and tfp installed.
    - Fallback: MC Dropout using Keras (if tensorflow installed) as approximate Bayesian NN.
    - Otherwise skip the deep probabilistic demo but still run GPR/GPC.

Outputs saved under day24_artifacts/.

Run:
    PowerShell with your .venv active
"""
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.datasets import load_diabetes, make_moons
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline

# Optional deep probabilistic dependencies
TF_AVAILABLE = False
TFP_AVAILABLE = False
try:
    import tensorflow as tf
    TF_AVAILABLE = True
    try:
        import tensorflow_probability as tfp
        TFP_AVAILABLE = True
    except Exception:
        TFP_AVAILABLE = False
except Exception:
    TF_AVAILABLE = False
    TFP_AVAILABLE = False

# --- Paths ---
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "day24_artifacts"
MODELS_DIR = OUT_DIR / "day24_models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

GPR_PLOT = OUT_DIR / "day24_gpr_diabetes.png"
GPC_PLOT = OUT_DIR / "day24_gpc_moons.png"
BNN_PLOT = OUT_DIR / "day24_bnn_predictions.png"
TFP_TRACE = OUT_DIR / "day24_tfp_trace.png"
METRICS_FILE = OUT_DIR / "day24_metrics.txt"

start_all = time.time()

# -------------------------
# 1) Gaussian Process Regression (GPR) on diabetes dataset (single feature view)
# -------------------------
print("Running GaussianProcessRegressor demo (diabetes dataset, single feature view)...")
diab = load_diabetes(as_frame=True)
X = diab.data
y = diab.target

# choose a single informative feature for clean plotting (bmi or bp)
feat = "bmi" if "bmi" in X.columns else X.columns[0]
X_feat = X[[feat]].values  # shape (n_samples, 1)

Xtr, Xte, ytr, yte = train_test_split(X_feat, y.values, test_size=0.25, random_state=42)

# Standardize feature for better GPR fitting
scaler = StandardScaler().fit(Xtr)
Xtr_s = scaler.transform(Xtr)
Xte_s = scaler.transform(Xte)

# kernel: constant * RBF + WhiteKernel (via alpha)
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-2, n_restarts_optimizer=5, random_state=42, normalize_y=True)
gpr.fit(Xtr_s, ytr)
# predict mean and std
y_mean, y_std = gpr.predict(Xte_s, return_std=True)

# Save GPR model
joblib.dump(gpr, MODELS_DIR / "gpr.joblib")

mse = mean_squared_error(yte, y_mean)
with open(METRICS_FILE, "w", encoding="utf-8") as f:
    f.write("Day24 - Probabilistic Deep Learning & GPs Metrics\n")
    f.write("="*60 + "\n\n")
    f.write(f"GPR (diabetes - feature: {feat})\n")
    f.write(f"  Test MSE: {mse:.4f}\n")
    f.write(f"  Mean predictive std: {np.mean(y_std):.4f}\n\n")

# Plot: scatter true vs predicted with uncertainty bands (sorted by feature)
order = np.argsort(Xte.flatten())
x_plot = Xte.flatten()[order]
x_plot_s = Xte_s.flatten()[order]
y_mean_p = y_mean[order]
y_std_p = y_std[order]
y_true_p = yte[order]

plt.figure(figsize=(8,6))
plt.scatter(x_plot, y_true_p, label="true", alpha=0.6)
plt.plot(x_plot, y_mean_p, label="GPR mean", color="C1")
plt.fill_between(x_plot, y_mean_p - 2*y_std_p, y_mean_p + 2*y_std_p, alpha=0.2, label="±2 std")
plt.xlabel(feat)
plt.ylabel("target")
plt.title("Day24 — GPR predictive mean ± uncertainty (diabetes)")
plt.legend()
plt.tight_layout()
plt.savefig(GPR_PLOT, dpi=150)
plt.close()
print(f"GPR plot saved to {GPR_PLOT}")

# -------------------------
# 2) Gaussian Process Classification (GPC) on make_moons toy dataset
# -------------------------
print("Running GaussianProcessClassifier demo (make_moons toy dataset)...")
Xm, ym = make_moons(n_samples=300, noise=0.2, random_state=42)
Xtr_m, Xte_m, ytr_m, yte_m = train_test_split(Xm, ym, test_size=0.25, random_state=42, stratify=ym)

# scale features for numerical stability
scaler_m = StandardScaler().fit(Xtr_m)
Xtr_m_s = scaler_m.transform(Xtr_m)
Xte_m_s = scaler_m.transform(Xte_m)

# GPC with an RBF kernel may be slow on many samples; use moderate settings
kernel_m = 1.0 * RBF(length_scale=1.0)
gpc = GaussianProcessClassifier(kernel=kernel_m, random_state=42, n_restarts_optimizer=2)
gpc.fit(Xtr_m_s, ytr_m)

# Save GPC
joblib.dump(gpc, MODELS_DIR / "gpc.joblib")

# Evaluate and produce probability surface
acc_gpc = accuracy_score(yte_m, gpc.predict(Xte_m_s))
proba_gpc = gpc.predict_proba(Xte_m_s)[:,1]
try:
    roc_gpc = roc_auc_score(yte_m, proba_gpc)
except Exception:
    roc_gpc = float("nan")

with open(METRICS_FILE, "a", encoding="utf-8") as f:
    f.write("GPC (make_moons toy):\n")
    f.write(f"  Test accuracy: {acc_gpc:.4f}\n")
    f.write(f"  Test ROC AUC: {roc_gpc:.4f}\n\n")

# Plot decision surface (probability)
xx, yy = np.meshgrid(np.linspace(Xm[:,0].min()-0.5, Xm[:,0].max()+0.5, 200),
                     np.linspace(Xm[:,1].min()-0.5, Xm[:,1].max()+0.5, 200))
grid = np.column_stack([xx.ravel(), yy.ravel()])
grid_s = scaler_m.transform(grid)
probs = gpc.predict_proba(grid_s)[:,1].reshape(xx.shape)

plt.figure(figsize=(8,6))
plt.contourf(xx, yy, probs, levels=25, cmap="RdBu", alpha=0.8)
plt.scatter(Xte_m[:,0], Xte_m[:,1], c=yte_m, edgecolor='k', cmap="RdBu", s=40)
plt.title("Day24 — GPC decision surface (predictive probability of class 1)")
plt.xlabel("X1"); plt.ylabel("X2")
plt.colorbar(label="P(class=1)")
plt.tight_layout()
plt.savefig(GPC_PLOT, dpi=150)
plt.close()
print(f"GPC plot saved to {GPC_PLOT}")

# -------------------------
# 3) Probabilistic Deep Learning
#    - Try TensorFlow Probability Bayesian NN (preferred)
#    - Fallback: MC Dropout using Keras (if TF installed)
# -------------------------
bnn_ran = False
if TFP_AVAILABLE:
    try:
        print("TensorFlow + TensorFlow Probability detected. Running small Bayesian NN demo (TFP)...")
        import tensorflow as tf
        import tensorflow_probability as tfp
        tfd = tfp.distributions
        tfk = tf.keras
        tfkl = tf.keras.layers
        tfpl = tfp.layers

        # Prepare data (breast_cancer classification, use a few features)
        from sklearn.datasets import load_breast_cancer
        bc = load_breast_cancer(as_frame=True)
        Xc = bc.data
        yc = bc.target
        # reduce dimensionality for quick demo: use first 8 features
        Xc_small = Xc.iloc[:, :8].values
        Xct, Xcv, yct, ycv = train_test_split(Xc_small, yc.values, test_size=0.25, random_state=42, stratify=yc)

        # scale
        scaler_b = StandardScaler().fit(Xct)
        Xct_s = scaler_b.transform(Xct)
        Xcv_s = scaler_b.transform(Xcv)

        # build small probabilistic NN with DenseVariational (Bayes by backprop)
        def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
            n = kernel_size + bias_size
            c = np.log(np.expm1(1.))
            return tf.keras.Sequential([
                tfpl.VariableLayer(2 * n, dtype=dtype),
                tfpl.DistributionLambda(lambda t: tfd.Independent(
                    tfd.Normal(loc=t[..., :n],
                               scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
                    reinterpreted_batch_ndims=1)),
            ])

        def prior_trainable(kernel_size, bias_size=0, dtype=None):
            n = kernel_size + bias_size
            return tf.keras.Sequential([
                tfpl.VariableLayer(n, dtype=dtype),
                tfpl.DistributionLambda(lambda t: tfd.Independent(tfd.Normal(loc=t, scale=1), reinterpreted_batch_ndims=1))
            ])

        inputs = tfkl.Input(shape=(Xct_s.shape[1],))
        x = tfkl.Dense(64, activation='relu')(inputs)
        # weight posterior/prior via DenseVariational (variational layers)
        x = tfpl.DenseVariational(32, posterior_mean_field, prior_trainable, activation="relu")(x)
        outputs = tfpl.DenseVariational(1, posterior_mean_field, prior_trainable, activation=None)(x)
        # outputs are distributions -> use sigmoid on mean for classification probability approximation
        model = tfk.Model(inputs=inputs, outputs=outputs)
        negloglik = lambda y, rv_y: -rv_y.log_prob(tf.expand_dims(y, -1))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=negloglik, metrics=[])

        # fit quickly (small epochs for demo)
        model.fit(Xct_s, yct, epochs=50, batch_size=32, verbose=0)

        # Predictive distribution: sample multiple forward passes by calling model(X, training=True) to keep stochasticity
        T = 50
        preds = []
        for t in range(T):
            rv = model(Xcv_s, training=True)  # distribution
            # get mean of distribution (rv.mean()) or sample
            try:
                m = rv.mean().numpy().flatten()
            except Exception:
                # fallback: sample
                m = rv.sample().numpy().flatten()
            preds.append(m)
        preds = np.vstack(preds)
        mean_pred = preds.mean(axis=0)
        std_pred = preds.std(axis=0)

        # evaluate
        y_pred_bin = (mean_pred >= 0.5).astype(int)
        acc_bnn = accuracy_score(ycv, y_pred_bin)
        try:
            roc_bnn = roc_auc_score(ycv, mean_pred)
        except Exception:
            roc_bnn = float("nan")

        with open(METRICS_FILE, "a", encoding="utf-8") as f:
            f.write("TFP Bayesian NN (small demo):\n")
            f.write(f"  val accuracy: {acc_bnn:.4f}\n")
            f.write(f"  val ROC AUC  : {roc_bnn:.4f}\n\n")

        # simple plot sorted by uncertainty
        ord_idx = np.argsort(std_pred)[::-1][:120]
        plt.figure(figsize=(12,5))
        plt.errorbar(np.arange(len(ord_idx)), mean_pred[ord_idx], yerr=std_pred[ord_idx], fmt='o', alpha=0.8)
        plt.title("Day24 — TFP Bayesian NN predictive mean ± std (subset sorted by uncertainty)")
        plt.xlabel("sample (subset)")
        plt.ylabel("predicted probability")
        plt.tight_layout()
        plt.savefig(BNN_PLOT, dpi=150)
        plt.close()

        # save model (TF models are large; save weights only)
        try:
            model.save(str(MODELS_DIR / "bnn_tfp_model"), save_format="tf")
        except Exception:
            # fallback: skip saving TF model
            pass

        with open(METRICS_FILE, "a", encoding="utf-8") as f:
            f.write(f"Saved TFP Bayesian NN plot: {BNN_PLOT}\n\n")
        print(f"TFP Bayesian NN demo done. Plot saved to {BNN_PLOT}")
        bnn_ran = True
    except Exception as e:
        print("TFP Bayesian NN demo failed or partially ran:", e)
        bnn_ran = False

elif TF_AVAILABLE:
    # fallback: MC Dropout with Keras
    try:
        print("TensorFlow detected but TensorFlow Probability not available. Running MC Dropout demo (Keras) as probabilistic NN fallback...")
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        from sklearn.datasets import load_breast_cancer
        bc = load_breast_cancer(as_frame=True)
        Xc = bc.data
        yc = bc.target
        Xc_small = Xc.iloc[:, :8].values
        Xct, Xcv, yct, ycv = train_test_split(Xc_small, yc.values, test_size=0.25, random_state=42, stratify=yc)
        scaler_b = StandardScaler().fit(Xct)
        Xct_s = scaler_b.transform(Xct)
        Xcv_s = scaler_b.transform(Xcv)

        def build_mc_dropout(input_dim, dropout_rate=0.3):
            inputs = keras.Input(shape=(input_dim,))
            x = layers.Dense(64, activation='relu')(inputs)
            x = layers.Dropout(dropout_rate)(x, training=True)  # ensure dropout active at training & inference if needed
            x = layers.Dense(32, activation='relu')(x)
            x = layers.Dropout(dropout_rate)(x, training=True)
            outputs = layers.Dense(1, activation='sigmoid')(x)
            model = keras.Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            return model

        mc_model = build_mc_dropout(Xct_s.shape[1], dropout_rate=0.3)
        mc_model.fit(Xct_s, yct, epochs=30, batch_size=32, verbose=0)

        # MC inference
        T = 50
        preds = []
        for t in range(T):
            p = mc_model(Xcv_s, training=True).numpy().flatten()
            preds.append(p)
        preds = np.vstack(preds)
        mean_pred = preds.mean(axis=0)
        std_pred = preds.std(axis=0)
        y_pred_bin = (mean_pred >= 0.5).astype(int)
        acc_mc = accuracy_score(ycv, y_pred_bin)
        try:
            roc_mc = roc_auc_score(ycv, mean_pred)
        except Exception:
            roc_mc = float("nan")

        with open(METRICS_FILE, "a", encoding="utf-8") as f:
            f.write("MC Dropout NN (Keras fallback):\n")
            f.write(f"  val accuracy: {acc_mc:.4f}\n")
            f.write(f"  val ROC AUC  : {roc_mc:.4f}\n\n")

        ord_idx = np.argsort(std_pred)[::-1][:120]
        plt.figure(figsize=(12,5))
        plt.errorbar(np.arange(len(ord_idx)), mean_pred[ord_idx], yerr=std_pred[ord_idx], fmt='o', alpha=0.8)
        plt.title("Day24 — MC Dropout predictive mean ± std (subset sorted by uncertainty)")
        plt.xlabel("sample (subset)")
        plt.ylabel("predicted probability")
        plt.tight_layout()
        plt.savefig(BNN_PLOT, dpi=150)
        plt.close()
        # Save Keras model weights
        try:
            mc_model.save(str(MODELS_DIR / "mc_dropout_keras_model"), save_format="tf")
        except Exception:
            pass
        with open(METRICS_FILE, "a", encoding="utf-8") as f:
            f.write(f"Saved MC Dropout plot: {BNN_PLOT}\n\n")
        print(f"MC Dropout demo done. Plot saved to {BNN_PLOT}")
        bnn_ran = True
    except Exception as e:
        print("MC Dropout demo failed:", e)
        bnn_ran = False
else:
    print("No TensorFlow detected — skipping probabilistic deep NN demos (TFP or MC Dropout). You can install TensorFlow (+TFP) to run these examples.")

end_all = time.time()
with open(METRICS_FILE, "a", encoding="utf-8") as f:
    f.write(f"Total script runtime (s): {end_all - start_all:.1f}\n")

print("\nAll done. Artifacts saved to:", OUT_DIR)
