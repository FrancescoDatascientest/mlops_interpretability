import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



def save_confusion_matrix(y_true, y_pred, labels, model_name, artifacts_dir, display_labels=None):
    
    display_labels = labels if display_labels is None else display_labels

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)

    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix - {model_name}")

    path = os.path.join(artifacts_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()

    return path