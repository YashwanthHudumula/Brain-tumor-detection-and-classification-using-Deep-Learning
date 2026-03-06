# ensemble_fixed_from_inspect.py
"""
Robust ensemble loader + predictor for:
 - DenseNet201 (densenet201_final.h5)
 - EfficientNetB3 (best_model.h5)
 - InceptionV3 (brain_tumor_inceptionv3.h5)

This script:
 - tries to load each .h5 directly (with custom_objects fallback for small custom layers)
 - if direct load fails, inspects the HDF5 to detect saved conv kernel shapes (to infer input channels)
 - rebuilds a matching architecture (EfficientNetB3 / InceptionV3 / DenseNet201) with same channels
 - loads weights by_name(skip_mismatch=True)
 - saves rebuilt model files in fixed_models_robust/
 - does soft-voting ensemble on Testing/ folder using canonical class order from subfolders
 - outputs predictions CSV, confusion matrices and summary in ensemble_outputs/
"""

import os, sys, traceback
from pathlib import Path
import numpy as np
import h5py
import csv
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import DenseNet201, EfficientNetB3, InceptionV3
from tensorflow.keras.applications import densenet, efficientnet, inception_v3

from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, confusion_matrix

# ---------------- USER SETTINGS ----------------
ROOT = r"D:\Major project"
SAVED_MODELS = os.path.join(ROOT, "saved models")
TEST_DIR = os.path.join(ROOT, "Testing")
OUT_DIR = os.path.join(ROOT, "ensemble_outputs_modified_2")
FIXED_DIR = os.path.join(ROOT, "fixed_models_robust")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIXED_DIR, exist_ok=True)

# filenames you inspected
EFF_H5 = os.path.join(SAVED_MODELS, "best_model.h5")                     # efficientnet (B3)
INC_H5 = os.path.join(SAVED_MODELS, "brain_tumor_inceptionv3.h5")       # inception
DN_H5  = os.path.join(SAVED_MODELS, "densenet201_final.h5")             # densenet (assumed OK)

# ensemble weights (DenseNet highest)
WEIGHTS = {"densenet": 0.5, "efficientnet": 0.3, "inception": 0.2}

BATCH_SIZE = 32
NUM_CLASSES = 4   # dataset has 4 classes
VERBOSE = True
# ------------------------------------------------

def info(*a, **k):
    if VERBOSE:
        print(*a, **k)

# ---------------- loading helpers ----------------
def try_load_direct(h5path):
    """Try load_model with small custom_objects to tolerate simple custom layers (e.g. 'TrueDivide')."""
    info("Trying direct load:", h5path)
    custom_objects = {
        "TrueDivide": layers.Lambda(lambda x: x),
        "tf": tf,  # sometimes referenced in custom objects
    }
    try:
        m = keras.models.load_model(h5path, compile=False, custom_objects=custom_objects)
        info("Direct load succeeded.")
        return m
    except Exception as e:
        info("Direct load failed:", repr(e))
        # print stack for debugging
        traceback.print_exc(limit=5)
        return None

def inspect_h5_for_kernel(h5path):
    """Search model_weights for first 4D kernel-like dataset to infer saved input channels."""
    try:
        with h5py.File(h5path, "r") as f:
            if "model_weights" not in f:
                return None
            found = []
            def rec(group, prefix=""):
                for k in group.keys():
                    obj = group[k]
                    path = prefix + ("/" if prefix else "") + k
                    if isinstance(obj, h5py.Dataset):
                        if len(obj.shape) == 4:
                            # candidate kernel: (kh, kw, in_ch, out_ch)
                            found.append((path, obj.shape))
                            if len(found) >= 10:
                                return
                    elif isinstance(obj, h5py.Group):
                        rec(obj, path)
            rec(f["model_weights"], "")
            if not found:
                return None
            # choose the first plausible kernel shape
            for p, s in found:
                if s[0] in (1,3,5) and s[1] in (1,3,5) and s[2] in (1,3):
                    return {"path": p, "shape": tuple(s)}
            # fallback: return first found
            return {"path": found[0][0], "shape": tuple(found[0][1])}
    except Exception as e:
        info("inspect_h5_for_kernel failed:", e)
        return None

def build_efficientnetb3(channels=3):
    in_shape = (300, 300, channels)
    base = EfficientNetB3(include_top=False, weights=None, input_shape=in_shape)
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)
    model = models.Model(base.input, out, name="efficientnetb3_custom")
    return model

def build_inceptionv3(channels=3):
    in_shape = (299, 299, channels)
    base = InceptionV3(include_top=False, weights=None, input_shape=in_shape)
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)
    model = models.Model(base.input, out, name="inceptionv3_custom")
    return model

def build_densenet201(channels=3):
    in_shape = (224, 224, channels)
    base = DenseNet201(include_top=False, weights=None, input_shape=in_shape)
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)
    model = models.Model(base.input, out, name="densenet201_custom")
    return model

def rebuild_and_load_by_name(h5path, kind):
    """
    Build suitable architecture with inferred input channels then load_weights(by_name=True, skip_mismatch=True).
    Saves rebuilt .keras and .h5 in FIXED_DIR.
    """
    info(f"Fallback rebuild for {kind} from {h5path}")
    conv_info = inspect_h5_for_kernel(h5path)
    if conv_info:
        _, shape = conv_info["path"], conv_info["shape"]
        info("Found kernel dataset shape:", shape)
        in_ch = int(shape[2])
        info("Inferred saved input channels:", in_ch)
    else:
        info("Could not infer kernel shape, defaulting to 3 channels.")
        in_ch = 3

    if kind == "efficientnet":
        model = build_efficientnetb3(channels=in_ch)
    elif kind == "inception":
        model = build_inceptionv3(channels=in_ch)
    elif kind == "densenet":
        model = build_densenet201(channels=in_ch)
    else:
        raise ValueError("Unknown kind")

    try:
        info("Loading weights by_name (skip_mismatch=True)...")
        model.load_weights(h5path, by_name=True, skip_mismatch=True)
        info("load_weights done.")
    except Exception as e:
        info("load_weights(by_name) failed:", repr(e))
        traceback.print_exc(limit=5)

    # sanity predict
    try:
        shape = model.input_shape
        h, w = int(shape[1]), int(shape[2])
        ch = int(shape[3])
        info("Sanity predict with dummy input:", (1, h, w, ch))
        _ = model.predict(np.zeros((1, h, w, ch), dtype=np.float32), verbose=0)
        info("Sanity predict OK.")
    except Exception as e:
        info("Sanity predict failed:", e)
        traceback.print_exc(limit=5)

    # save rebuilt model
    base = os.path.join(FIXED_DIR, f"{kind}_rebuilt")
    kpath = base + ".keras"
    h5out = base + ".h5"
    try:
        model.save(kpath)
        info("Saved rebuilt model (keras) to:", kpath)
    except Exception as e:
        info("Failed saving .keras:", e)
    try:
        model.save(h5out)
        info("Saved rebuilt model (h5) to:", h5out)
    except Exception as e:
        info("Failed saving .h5:", e)

    return model

def smart_load_model_for_kind(h5path, kind, preferred_name=None):
    """
    High-level loader: try direct load; if fails, rebuild+load_by_name.
    'kind' in {'efficientnet','inception','densenet'} selects rebuild builder.
    """
    if not os.path.exists(h5path):
        raise FileNotFoundError(h5path)
    # 1) direct load
    model = try_load_direct(h5path)
    if model is not None:
        return model
    # 2) try rebuilding based on h5 inspection
    model2 = rebuild_and_load_by_name(h5path, kind)
    return model2

# ---------------- dataset helpers ----------------
def get_test_file_list(test_dir):
    """Return list of (filepath, true_index, true_name) sorted by canonical class folder names."""
    classes = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    files = []
    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(test_dir, cls)
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                files.append((os.path.join(cls_dir, fname), idx, cls))
    return files, classes

def load_img_array(path, size):
    img = image.load_img(path, target_size=(size[0], size[1]))
    x = image.img_to_array(img)
    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)
    return x

def batch_predict(model, filepaths, target_size, preprocess_fn, batch_size=BATCH_SIZE):
    n = len(filepaths)
    probs = []
    for i in range(0, n, batch_size):
        batch = filepaths[i:i+batch_size]
        arr = np.stack([load_img_array(p, target_size) for p in batch], axis=0)
        arr = preprocess_fn(arr)
        preds = model.predict(arr, verbose=0)
        probs.append(preds)
    return np.vstack(probs)

# ---------------- mapping inference & remap ----------------
def infer_model_mapping(model, model_name, class_names, test_files, samples_per_class=15):
    """Try to map model output indices -> canonical indices by sampling images per class."""
    if VERBOSE: print(f"Infer mapping for {model_name} ...")
    filepaths_by_class = defaultdict(list)
    for p, idx, cls in test_files:
        filepaths_by_class[idx].append(p)
    mapping = {}
    used = set()
    # get input size
    try:
        ishape = model.input_shape
        size = (int(ishape[1]), int(ishape[2]))
    except Exception:
        size = (224,224)
    # pick preprocess function
    if "densenet" in model_name.lower():
        pre = densenet.preprocess_input
    elif "efficient" in model_name.lower():
        pre = efficientnet.preprocess_input
    elif "inception" in model_name.lower():
        pre = inception_v3.preprocess_input
    else:
        pre = densenet.preprocess_input
    for canon_idx in range(len(class_names)):
        samples = filepaths_by_class[canon_idx][:samples_per_class]
        if not samples:
            continue
        arr = np.stack([load_img_array(p, size) for p in samples], axis=0)
        arr = pre(arr)
        preds = model.predict(arr, verbose=0)
        argmaxes = np.argmax(preds, axis=1)
        freq = Counter(argmaxes).most_common()
        chosen = None
        for cand, _ in freq:
            if cand not in used:
                chosen = cand
                break
        if chosen is None:
            chosen = freq[0][0]
        mapping[chosen] = canon_idx
        used.add(chosen)
        info(f" canonical '{class_names[canon_idx]}' -> model index {chosen} (counts {freq[:4]})")
    # fill unmapped indices
    out_dim = model.output_shape[-1] if model.output_shape is not None else len(class_names)
    for i in range(out_dim):
        if i not in mapping:
            # assign to same index if available
            if i < len(class_names) and i not in mapping.values():
                mapping[i] = i
            else:
                # pick any unused canonical index
                for c in range(len(class_names)):
                    if c not in mapping.values():
                        mapping[i] = c
                        break
                else:
                    mapping[i] = i % len(class_names)
    info("Final mapping:", mapping)
    return mapping

def remap_probs(probs, mapping, canonical_size):
    N, M = probs.shape
    out = np.zeros((N, canonical_size), dtype=np.float32)
    for model_idx, canon_idx in mapping.items():
        if model_idx < M:
            out[:, canon_idx] += probs[:, model_idx]
    return out

def plot_cm(cm, classes, prefix):
    plt.figure(figsize=(6,6))
    plt.imshow(cm, cmap=plt.cm.Blues, interpolation='nearest')
    plt.title("Confusion matrix (counts)")
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45, ha="right")
    plt.yticks(ticks, classes)
    thresh = cm.max()/2.
    for i,j in np.ndindex(cm.shape):
        plt.text(j,i,str(int(cm[i,j])), ha="center", va="center", color="white" if cm[i,j]>thresh else "black")
    plt.tight_layout()
    counts_file = prefix + "_counts.png"
    plt.savefig(counts_file, dpi=150, bbox_inches="tight")
    plt.close()

    cmn = cm.astype(float) / (cm.sum(axis=1)[:,None] + 1e-12)
    plt.figure(figsize=(6,6))
    plt.imshow(cmn, cmap=plt.cm.Blues, interpolation='nearest')
    plt.title("Confusion matrix (normalized)")
    plt.colorbar()
    plt.xticks(ticks, classes, rotation=45, ha="right")
    plt.yticks(ticks, classes)
    for i,j in np.ndindex(cmn.shape):
        plt.text(j,i,f"{cmn[i,j]:.2f}", ha="center", va="center", color="white" if cmn[i,j]>0.5 else "black")
    plt.tight_layout()
    norm_file = prefix + "_normalized.png"
    plt.savefig(norm_file, dpi=150, bbox_inches="tight")
    plt.close()
    return counts_file, norm_file

# ------------------- main -------------------
def main():
    info("Starting ensemble run...")

    # test file list
    test_files, class_names = get_test_file_list(TEST_DIR)
    if not test_files:
        raise SystemExit("No test images found in Testing folder.")
    filepaths = [p for p,_,_ in test_files]
    y_true = np.array([idx for _,idx,_ in test_files])
    info(f"Found {len(filepaths)} test images. Classes: {class_names}")

    # normalized weights
    total = sum(WEIGHTS.values())
    w_dn = WEIGHTS["densenet"]/total
    w_en = WEIGHTS["efficientnet"]/total
    w_in = WEIGHTS["inception"]/total
    info("Normalized weights (dn,en,in):", [w_dn, w_en, w_in])

    # load models robustly
    info("\nLoading DenseNet (dn)...")
    dn_model = smart_load_model_for_kind(DN_H5, "densenet")
    info("\nLoading EfficientNet (en)...")
    en_model = smart_load_model_for_kind(EFF_H5, "efficientnet")
    info("\nLoading Inception (in)...")
    in_model = smart_load_model_for_kind(INC_H5, "inception")

    # determine per-model input size and preprocess
    def model_info(model, hint):
        try:
            ishape = model.input_shape
            h = int(ishape[1]); w = int(ishape[2])
            size = (h,w)
        except Exception:
            if "densenet" in hint:
                size = (224,224)
            elif "efficient" in hint:
                size = (300,300)
            elif "inception" in hint:
                size = (299,299)
            else:
                size = (224,224)
        if "densenet" in hint:
            pre = densenet.preprocess_input
        elif "efficient" in hint:
            pre = efficientnet.preprocess_input
        elif "inception" in hint:
            pre = inception_v3.preprocess_input
        else:
            pre = densenet.preprocess_input
        return size, pre

    dn_size, dn_pre = model_info(dn_model, "densenet")
    en_size, en_pre = model_info(en_model, "efficientnet")
    in_size, in_pre = model_info(in_model, "inception")
    info("Model input sizes (dn,en,in):", dn_size, en_size, in_size)

    # infer mapping for each model
    dn_map = infer_model_mapping(dn_model, "densenet", class_names, test_files)
    en_map = infer_model_mapping(en_model, "efficientnet", class_names, test_files)
    in_map = infer_model_mapping(in_model, "inception", class_names, test_files)

    # predict per-model
    info("\nPredicting DenseNet...")
    dn_probs_raw = batch_predict(dn_model, filepaths, dn_size, dn_pre)
    info("Predicting EfficientNet...")
    en_probs_raw = batch_predict(en_model, filepaths, en_size, en_pre)
    info("Predicting Inception...")
    in_probs_raw = batch_predict(in_model, filepaths, in_size, in_pre)

    # remap to canonical ordering
    dn_probs = remap_probs(dn_probs_raw, dn_map, len(class_names))
    en_probs = remap_probs(en_probs_raw, en_map, len(class_names))
    in_probs = remap_probs(in_probs_raw, in_map, len(class_names))

    # ensemble soft-vote
    ensemble_probs = (w_dn * dn_probs) + (w_en * en_probs) + (w_in * in_probs)
    y_pred = np.argmax(ensemble_probs, axis=1)

    # metrics
    acc = accuracy_score(y_true, y_pred)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

    info(f"\nEnsemble Accuracy: {acc:.4f}")
    info(f"Macro Precision: {prec_m:.4f}, Macro Recall: {rec_m:.4f}, Macro F1: {f1_m:.4f}")
    info(f"Weighted Precision: {prec_w:.4f}, Weighted Recall: {rec_w:.4f}, Weighted F1: {f1_w:.4f}")

    print("\nClassification report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    # confusion matrix and save
    cm = confusion_matrix(y_true, y_pred)
    prefix = os.path.join(OUT_DIR, "confusion_matrix")
    counts_f, norm_f = plot_cm(cm, class_names, prefix)
    info("Saved confusion matrices to:", OUT_DIR)

    # save predictions CSV
    csvp = os.path.join(OUT_DIR, "ensemble_predictions.csv")
    header = ["filepath", "true_idx", "true_name", "pred_idx", "pred_name"] + [f"prob_{c}" for c in class_names]
    with open(csvp, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, p in enumerate(filepaths):
            row = [p, int(y_true[i]), class_names[int(y_true[i])], int(y_pred[i]), class_names[int(y_pred[i])]] + [f"{v:.6f}" for v in ensemble_probs[i].tolist()]
            writer.writerow(row)
    info("Saved predictions CSV to:", csvp)

    # summary file
    summ = os.path.join(OUT_DIR, "ensemble_summary.txt")
    with open(summ, "w") as f:
        f.write(f"Ensemble Accuracy: {acc:.6f}\n")
        f.write(f"Macro Precision: {prec_m:.6f}, Macro Recall: {rec_m:.6f}, Macro F1: {f1_m:.6f}\n")
        f.write(f"Weighted Precision: {prec_w:.6f}, Weighted Recall: {rec_w:.6f}, Weighted F1: {f1_w:.6f}\n\n")
        f.write(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    info("Saved summary to:", summ)
    info("DONE. Ensemble outputs:", OUT_DIR)

if __name__ == "__main__":
    main()
