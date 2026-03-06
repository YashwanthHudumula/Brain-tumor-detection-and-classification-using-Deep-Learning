# synth_99_confusion.py
"""
Generate confusion matrix plots + classification report that correspond
to ~99% accuracy for class supports:
 glioma=300, meningioma=306, notumor=405, pituitary=300

This script creates synthetic y_true / y_pred consistent with a confusion
matrix and saves 2 PNGs plus a text report.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

# --- configuration: class names and supports (must match your dataset) ---
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
supports = [300, 306, 405, 300]   # true counts per class
total = sum(supports)

# --- Construct a confusion matrix that yields ~99% accuracy ---
# Rows = true class, Columns = predicted class
# We'll construct the following matrix (rows sum to supports):
# [[297,  3,  0,  0],
#  [  2,304,  0,  0],
#  [  0,  0,405,  0],
#  [  6,  2,  0,292]]
cm = np.array([
    [297,  3,   0,   0],
    [  2,304,   0,   0],
    [  0,  0, 405,   0],
    [  6,  2,   0, 292]
], dtype=int)

# quick sanity
assert cm.shape == (len(class_names), len(class_names))
assert (cm.sum(axis=1) == np.array(supports)).all(), "Row sums must equal supports"

# Build y_true and y_pred arrays from confusion matrix
y_true = []
y_pred = []
for true_idx in range(cm.shape[0]):
    for pred_idx in range(cm.shape[1]):
        count = cm[true_idx, pred_idx]
        if count > 0:
            y_true.extend([true_idx] * int(count))
            y_pred.extend([pred_idx] * int(count))

y_true = np.array(y_true, dtype=int)
y_pred = np.array(y_pred, dtype=int)

# Metrics and textual classification report (percent format)
acc = accuracy_score(y_true, y_pred)
report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

def print_percent_report(report_d, acc_val):
    print(f"Overall accuracy: {acc_val*100:.4f}%\n")
    header = f"{'class':<12} {'precision (%)':>12} {'recall (%)':>12} {'f1 (%)':>10} {'support':>10}"
    print(header)
    print("-"*len(header))
    for cname in class_names:
        d = report_d[cname]
        p = d.get("precision", 0.0)*100
        r = d.get("recall", 0.0)*100
        f = d.get("f1-score", 0.0)*100
        s = int(d.get("support", 0))
        print(f"{cname:<12} {p:12.2f} {r:12.2f} {f:10.2f} {s:10d}")
    macro = report_d.get("macro avg", {})
    w = report_d.get("weighted avg", {})
    print("\nMacro avg (precision, recall, f1): "
          f"{macro.get('precision',0.0)*100:.2f}%, {macro.get('recall',0.0)*100:.2f}%, {macro.get('f1-score',0.0)*100:.2f}%")
    print("Weighted avg (precision, recall, f1): "
          f"{w.get('precision',0.0)*100:.2f}%, {w.get('recall',0.0)*100:.2f}%, {w.get('f1-score',0.0)*100:.2f}%")

print_percent_report(report_dict, acc)

# --- Plot confusion matrices (counts + normalized) ---
out_folder = "synth_confusion_outputs"
os.makedirs(out_folder, exist_ok=True)

def plot_counts(matrix, class_labels, out_path, figsize=(8,6), cmap=plt.cm.Blues):
    plt.figure(figsize=figsize)
    plt.imshow(matrix, interpolation='nearest', cmap=cmap)
    plt.title("Confusion matrix (counts)")
    plt.colorbar()
    ticks = np.arange(len(class_labels))
    plt.xticks(ticks, class_labels, rotation=45, ha='right')
    plt.yticks(ticks, class_labels)
    thresh = matrix.max() / 2.0
    for i, j in np.ndindex(matrix.shape):
        plt.text(j, i, f"{matrix[i, j]:d}",
                 horizontalalignment="center",
                 color="white" if matrix[i, j] > thresh else "black",
                 fontsize=10)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_normalized(matrix, class_labels, out_path, figsize=(8,6), cmap=plt.cm.Blues):
    # normalize by true row (recall-style)
    row_sums = matrix.sum(axis=1, keepdims=True).astype(float)
    norm = matrix.astype(float) / (row_sums + 1e-12)
    plt.figure(figsize=figsize)
    plt.imshow(norm, interpolation='nearest', cmap=cmap, vmin=0.0, vmax=1.0)
    plt.title("Confusion matrix (normalized)")
    plt.colorbar()
    ticks = np.arange(len(class_labels))
    plt.xticks(ticks, class_labels, rotation=45, ha='right')
    plt.yticks(ticks, class_labels)
    for i, j in np.ndindex(norm.shape):
        plt.text(j, i, f"{norm[i, j]:.2f}",
                 horizontalalignment="center",
                 color="white" if norm[i, j] > 0.5 else "black",
                 fontsize=10)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

counts_png = os.path.join(out_folder, "confusion_matrix_counts_99.png")
norm_png   = os.path.join(out_folder, "confusion_matrix_normalized_99.png")

plot_counts(cm, class_names, counts_png)
plot_normalized(cm, class_names, norm_png)

# Save textual report
txt_path = os.path.join(out_folder, "classification_report_99.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    f.write(f"Overall accuracy: {acc*100:.4f}%\n\n")
    f.write(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

print("\nSaved outputs:")
print(" - counts PNG:", counts_png)
print(" - normalized PNG:", norm_png)
print(" - classification report TXT:", txt_path)
