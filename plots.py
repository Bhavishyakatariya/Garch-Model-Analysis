import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from scipy import stats
import shap

# ==============================
# 1. RETURNS & VOL CLUSTERING
# ==============================
def plot_returns_volatility(df):
    returns = df['returns'].dropna()

    plt.figure(figsize=(12,6))
    plt.plot(returns, label="Returns")
    plt.plot(returns**2, label="Squared Returns", alpha=0.6)
    plt.title("Volatility Clustering")
    plt.legend()
    plt.savefig("returns_volatility.png", dpi=300)
    plt.close()


# ==============================
# 2. GARCH VOLATILITY
# ==============================
def plot_garch_volatility(returns, vol):
    plt.figure(figsize=(12,6))
    plt.plot(returns, label="Returns", alpha=0.5)
    plt.plot(vol, label="Volatility")
    plt.title("Conditional Volatility")
    plt.legend()
    plt.savefig("garch_volatility.png", dpi=300)
    plt.close()


# ==============================
# 3. MULTI GARCH COMPARISON
# ==============================
def plot_garch_family(egarch, gjr, exo):
    plt.figure(figsize=(12,6))
    plt.plot(egarch, label="EGARCH")
    plt.plot(gjr, label="GJR-GARCH")
    plt.plot(exo, label="EGARCH-X")
    plt.title("GARCH Family Comparison")
    plt.legend()
    plt.savefig("garch_family.png", dpi=300)
    plt.close()


# ==============================
# 4. RESIDUAL DIAGNOSTICS
# ==============================
def plot_residuals(residuals):
    fig, ax = plt.subplots(3,1, figsize=(10,12))

    ax[0].plot(residuals)
    ax[0].set_title("Standardized Residuals")

    plot_acf(residuals, ax=ax[1], lags=40)
    ax[1].set_title("ACF Residuals")

    plot_acf(residuals**2, ax=ax[2], lags=40)
    ax[2].set_title("ACF Squared Residuals")

    plt.tight_layout()
    plt.savefig("residual_diagnostics.png", dpi=300)
    plt.close()


# ==============================
# 5. DISTRIBUTION CHECK
# ==============================
def plot_distribution(residuals):
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    sns.histplot(residuals, kde=True)
    plt.title("Residual Distribution")

    plt.subplot(1,2,2)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("Q-Q Plot")

    plt.tight_layout()
    plt.savefig("distribution.png", dpi=300)
    plt.close()


# ==============================
# 6. XGBOOST VS ACTUAL
# ==============================
def plot_xgb_vs_actual(actual, pred):
    plt.figure(figsize=(12,6))
    plt.plot(actual, label="Actual Volatility")
    plt.plot(pred, label="XGBoost Prediction", linestyle='dashed')
    plt.title("XGBoost vs Actual Volatility")
    plt.legend()
    plt.savefig("xgb_vs_actual.png", dpi=300)
    plt.close()


# ==============================
# 7. ERROR COMPARISON
# ==============================
def plot_error(actual, garch, xgb):
    plt.figure(figsize=(12,6))
    plt.plot(abs(actual - garch), label="GARCH Error")
    plt.plot(abs(actual - xgb), label="XGBoost Error")
    plt.title("Error Comparison")
    plt.legend()
    plt.savefig("error_comparison.png", dpi=300)
    plt.close()


# ==============================
# 8. FEATURE IMPORTANCE
# ==============================
def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_

    plt.figure(figsize=(10,6))
    sns.barplot(x=importance, y=feature_names)
    plt.title("XGBoost Feature Importance")
    plt.savefig("feature_importance.png", dpi=300)
    plt.close()


# ==============================
# 9. ROLLING VOLATILITY
# ==============================
def plot_rolling_volatility(returns, window=30):
    rolling_vol = returns.rolling(window).std()

    plt.figure(figsize=(12,6))
    plt.plot(rolling_vol)
    plt.title(f"Rolling Volatility ({window})")
    plt.savefig("rolling_volatility.png", dpi=300)
    plt.close()


# ==============================
# 10. LEVERAGE EFFECT (EGARCH)
# ==============================
def plot_leverage_effect(returns, volatility):
    returns = returns.dropna()
    volatility = volatility.reindex(returns.index)

    plt.figure(figsize=(8,6))
    plt.scatter(returns, volatility, alpha=0.4)

    plt.axvline(0, linestyle='--')
    plt.title("Leverage Effect (EGARCH)")
    plt.xlabel("Returns")
    plt.ylabel("Volatility")

    plt.savefig("leverage_effect.png", dpi=300)
    plt.close()


# ==============================
# 11. SHAP SUMMARY
# ==============================
def plot_shap_summary(model, X, feature_names):
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.savefig("shap_summary.png", dpi=300, bbox_inches='tight')
    plt.close()


# ==============================
# 12. SHAP BAR
# ==============================
def plot_shap_bar(model, X, feature_names):
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    shap.plots.bar(shap_values, show=False)
    plt.savefig("shap_bar.png", dpi=300, bbox_inches='tight')
    plt.close()