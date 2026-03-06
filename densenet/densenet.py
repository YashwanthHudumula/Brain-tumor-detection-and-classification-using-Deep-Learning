# gradcam_occlusion_fixed.py
import os
import sys
import pathlib
import random
import traceback
import argparse
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import pandas as pd

# ---------- USER SETTINGS ----------
ROOT = r"D:\Major project\densenet"
TEST_DIR = os.path.join(ROOT, "Testing")
MODEL_PATH = os.path.join(ROOT, "densenet201_final.h5")
OUT_DIR = os.path.join(ROOT, "gradcam_occlusion_outputs")
IMG_SIZE = (224, 224)
PATCH_SIZE = 32      # occlusion square size (try 24-48)
STRIDE = 16          # step between patches
NUM_SAMPLE = None    # None -> all selected images; or an int to limit
ONLY_MISCLASSIFIED = True
ALPHA = 0.45
BATCH_PRED = 8       # predict in small batches for speed
# -----------------------------------

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
ERROR_LOG = os.path.join(ROOT, "gradcam_errors.log")

def find_class_names(test_dir):
    return sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])

def list_all_test_images(test_dir, class_names):
    files = []
    for cls in class_names:
        cls_dir = os.path.join(test_dir, cls)
        if not os.path.exists(cls_dir):
            continue
        for root, _, filenames in os.walk(cls_dir):
            for f in sorted(filenames):
                if f.lower().endswith(EXTS):
                    files.append((os.path.join(root, f), cls))
    return files

def load_and_preprocess(img_path, target_size):
    img = Image.open(img_path).convert("RGB")
    img_resized = img.resize(target_size, Image.BILINEAR)
    arr = np.array(img_resized).astype(np.float32)
    arr = preprocess_input(arr)   # DenseNet preprocessing (expects 0..255 floats)
    # return both preprocessed arr (H,W,3) and the RGB PIL image for overlay
    return arr, img_resized

def select_images(class_names):
    csv_path = os.path.join(ROOT, "plots", "test_predictions.csv")
    if ONLY_MISCLASSIFIED and os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if {"filename","true_label","pred_label"}.issubset(df.columns):
                mis = df[df["true_label"] != df["pred_label"]]
                if len(mis) > 0:
                    print(f"Using {len(mis)} misclassified examples from {csv_path}")
                    return [(str(r["filename"]), r["true_label"]) for _, r in mis.iterrows()]
        except Exception as e:
            print("Could not read CSV for misclassified filter:", repr(e))
    all_files = list_all_test_images(TEST_DIR, class_names)
    if NUM_SAMPLE is None:
        return all_files
    random.shuffle(all_files)
    return all_files[:NUM_SAMPLE]

def occlusion_heatmap_single(model, img_arr, pred_index, patch_size=PATCH_SIZE, stride=STRIDE, batch_pred=BATCH_PRED):
    """
    img_arr: (H,W,3) preprocessed float32 (not batched)
    returns heatmap HxW with importance (drop in probability when occluded)
    """
    H, W, C = img_arr.shape
    # baseline prediction
    orig = np.expand_dims(img_arr, axis=0)
    base_pred = model.predict(orig, verbose=0)[0, pred_index]

    heatmap = np.zeros((H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.int32)

    # compute fill value: mean of image (in preprocessed space)
    fill = np.mean(img_arr, axis=(0,1), keepdims=True)  # shape (1,1,3)

    patches = []
    coords = []

    def flush_batch(patches, coords):
        if not patches:
            return
        batch = np.stack(patches, axis=0)
        preds = model.predict(batch, verbose=0)
        for i, (yy1,xx1,yy2,xx2) in enumerate(coords):
            drop = float(base_pred - preds[i, pred_index])
            if drop < 0:
                drop = 0.0
            heatmap[yy1:yy2, xx1:xx2] += drop
            counts[yy1:yy2, xx1:xx2] += 1
        return [], []

    # slide window
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y1 = y
            x1 = x
            y2 = min(H, y1 + patch_size)
            x2 = min(W, x1 + patch_size)
            im_copy = img_arr.copy()
            im_copy[y1:y2, x1:x2, :] = fill
            patches.append(im_copy)
            coords.append((y1,x1,y2,x2))
            if len(patches) >= batch_pred:
                patches, coords = flush_batch(patches, coords)

    # final flush
    if patches:
        patches, coords = flush_batch(patches, coords)

    # avoid division by zero
    counts = np.maximum(counts, 1)
    heatmap = heatmap / counts
    # normalize 0..1
    heatmap = np.maximum(heatmap, 0)
    denom = heatmap.max() if heatmap.max() != 0 else 1e-10
    heatmap /= denom
    return heatmap

def overlay_heatmap(original_pil, heatmap, outpath, alpha=ALPHA):
    heatmap_img = Image.fromarray(np.uint8(cm.jet(heatmap) * 255))
    heatmap_img = heatmap_img.resize(original_pil.size, resample=Image.BICUBIC)
    blended = Image.blend(original_pil.convert("RGB"), heatmap_img.convert("RGB"), alpha=alpha)
    blended.save(outpath)

def main(debug_one=None):
    # clear / create error log
    with open(ERROR_LOG, "w", encoding="utf-8") as ef:
        ef.write("Grad-CAM occlusion error log\n")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found: " + MODEL_PATH)
    print("Loading model:", MODEL_PATH)
    model = load_model(MODEL_PATH, compile=False)
    class_names = find_class_names(TEST_DIR)
    if not class_names:
        raise RuntimeError("No class folders found under Testing/")
    print("Classes:", class_names)

    in_shape = model.input_shape
    target_size = (in_shape[1] or IMG_SIZE[0], in_shape[2] or IMG_SIZE[1])
    print("Model input size:", target_size)

    if debug_one:
        to_process = [(debug_one, "debug")]
    else:
        to_process = select_images(class_names)
    print("Selected", len(to_process), "images.")

    os.makedirs(OUT_DIR, exist_ok=True)
    succ = 0
    fail = 0

    for (img_path, true_label) in to_process:
        try:
            arr, pil_img = load_and_preprocess(img_path, target_size)
            # arr shape either (H,W,3) or (1,H,W,3)
            if arr.ndim == 4:
                arr = arr[0]
            # get base prediction
            preds = model.predict(np.expand_dims(arr, axis=0), verbose=0)
            pred_index = int(np.argmax(preds[0]))
            pred_label = class_names[pred_index]
            # compute occlusion heatmap
            heatmap = occlusion_heatmap_single(model, arr, pred_index, patch_size=PATCH_SIZE, stride=STRIDE, batch_pred=BATCH_PRED)
            base = pathlib.Path(img_path).stem
            out_fname = f"{base}__pred-{pred_label}__true-{true_label}.png"
            out_path = os.path.join(OUT_DIR, out_fname)
            overlay_heatmap(pil_img.resize(target_size), heatmap, out_path)
            print("Saved overlay:", out_path)
            succ += 1
        except Exception as e:
            fail += 1
            tb = traceback.format_exc()
            print(f"Failed on {img_path} : {repr(e)}")
            with open(ERROR_LOG, "a", encoding="utf-8") as ef:
                ef.write(f"\nFailed on {img_path} : {repr(e)}\n")
                ef.write(tb + "\n")
            # If debugging single image, re-raise so you see traceback in console
            if debug_one:
                raise
            # continue to next image
            continue

    print(f"Done. succeeded: {succ} failed: {fail} outputs in: {OUT_DIR}")
    print("If failures occurred, see", ERROR_LOG)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug_one", type=str, default=None, help="Path to single image to debug")
    args = parser.parse_args()
    main(debug_one=args.debug_one)
