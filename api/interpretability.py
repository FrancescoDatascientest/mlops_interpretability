import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def init_explainer(model):
    """
    Crée un explainer SHAP à partir du modèle
    """
    explainer = shap.TreeExplainer(model)
    return explainer


def compute_local_shap(explainer, X_processed: pd.DataFrame):
    """
    Retourne les SHAP values locales pour X_processed
    """
    shap_values = explainer.shap_values(X_processed, approximate=True)
    return shap_values  # garder en numpy array


def generate_summary_plot(X, shap_values, output_path="summary_plot.png"):
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values[:, :, 3], X, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def generate_dependence_plot(X, shap_values, feature, output_path="dependence_plot.png"):
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(feature, shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def generate_force_plot(explainer, shap_values, X, index=0, output_path="force_plot.html"):
    force_html = shap.force_plot(
        explainer.expected_value,
        shap_values[index, :],
        X.iloc[index, :],
        matplotlib=False
    )
    shap.save_html(output_path, force_html)
    return output_path