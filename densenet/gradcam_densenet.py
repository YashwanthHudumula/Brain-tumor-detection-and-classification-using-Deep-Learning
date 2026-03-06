# gradcam_backend_fallback.py
"""
Robust Grad-CAM for saved Keras full models.
Tries eager GradientTape first; if that fails, uses symbolic K.backend function
so it will work with saved Functional models that don't accept eager tensors inside tape.

Outputs saved to: gradcam_outputs_final_backend/
"""
import os, pathlib, random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import pandas as pd
from tensorflow.keras import backend as K

# ---------- USER SETTINGS ----------
ROOT = r"D:\Major project\densenet"
TEST_DIR = os.path.join(ROOT, "Testing")
MODEL_PATH = os.path.join(ROOT, "densenet201_final.h5")
OUT_DIR = os.path.join(ROOT, "gradcam_outputs_final_backend")
ALPHA = 0.45
SEED = 42
ONLY_MISCLASSIFIED = True   # use plots/test_predictions.csv if available
MAX_IMAGES = None           # None => all selected images; set int to sample
# -----------------------------------

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

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
    arr = preprocess_input(arr)   # DenseNet preprocess
    arr = np.expand_dims(arr, axis=0)
    return arr, img

def find_last_conv_layer_recursive(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
        if isinstance(layer, tf.keras.Model) or hasattr(layer, "layers"):
            try:
                sub = find_last_conv_layer_recursive(layer)
                if sub is not None:
                    return sub
            except Exception:
                pass
    return None

def gradcam_via_eager(model, conv_layer_obj, img_array, pred_index=None):
    """Eager tape method (fast, readable). May fail on some saved models."""
    conv_model = tf.keras.models.Model(inputs=model.inputs, outputs=conv_layer_obj.output)
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs = conv_model(img_tensor)
        tape.watch(conv_outputs)
        preds = model(img_tensor)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        raise RuntimeError("Eager grads are None")
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0].numpy()
    pooled = pooled_grads.numpy()
    # weighted sum
    for i in range(pooled.shape[-1]):
        conv_outputs[..., i] *= pooled[i]
    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (heatmap.max() if heatmap.max() != 0 else 1e-10)
    return heatmap, preds.numpy()

def gradcam_via_symbolic(model, conv_layer_obj, img_array, pred_index=None):
    """
    Symbolic backend approach: construct Keras graph expressions and a K.function
    This works reliably with loaded Functional models because it uses the model's
    symbolic tensors rather than constructing a new eager model.
    """
    # symbolic tensors from model graph
    model_input = model.input                             # KerasTensor
    conv_output = conv_layer_obj.output                   # symbolic tensor
    model_output = model.output                           # symbolic tensor

    # we'll compute gradients of the predicted class wrt conv_output
    # create placeholder for index by computing grads for each class then pick one at runtime
    # but easier: compute grads symbolic for class index 0..n by slicing at runtime
    # so build function that returns (pooled_grads, conv_output_val, preds)
    # Step 1: create symbolic class scalar - we will compute grads for all classes and select index later.
    # Simpler: compute grads for predicted argmax (but argmax is non-diff). So instead we'll return full preds and
    # return conv and grads for each class by computing gradients per class in python per image.
    # Implementation: create func that returns conv_output and model_output and gradients w.r.t conv_output
    # for each class via K.gradients (one at a time) — then call per image for the target class.
    preds = model_output
    num_classes = int(preds.shape[-1])
    if num_classes is None:
        # fallback to running model.predict once to get shape
        tmp = model.predict(img_array)
        num_classes = tmp.shape[-1]

    # Build function returning conv_output and preds (we'll compute grads for chosen class via K.gradients)
    func = K.function([model_input], [conv_output, model_output])

    # call to get conv_val and pred vector
    conv_val, pred_val = func([img_array])
    if pred_index is None:
        pred_index = int(np.argmax(pred_val[0]))

    # compute symbolic gradient for that class
    y_c = model_output[:, pred_index]
    grads = K.gradients(y_c, conv_output)[0]  # symbolic gradient tensor
    # make function to fetch pooled_grads and conv_output value
    pooled_fn = K.function([model_input], [K.mean(grads, axis=(0,1,2)), conv_output, model_output])

    pooled_grads_val, conv_val2, pred_val2 = pooled_fn([img_array])
    pooled_grads_val = np.squeeze(pooled_grads_val)
    conv_val2 = np.squeeze(conv_val2)  # Hc,Wc,C
    # weight conv maps
    for i in range(pooled_grads_val.shape[-1]):
        conv_val2[..., i] *= pooled_grads_val[i]
    heatmap = np.mean(conv_val2, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (heatmap.max() if heatmap.max() != 0 else 1e-10)
    return heatmap, pred_val2

def overlay_heatmap_on_image(original_pil, heatmap, output_path, alpha=ALPHA):
    orig_w, orig_h = original_pil.size
    heatmap_img = Image.fromarray(np.uint8(cm.jet(heatmap) * 255))
    heatmap_img = heatmap_img.resize((orig_w, orig_h), resample=Image.BICUBIC)
    blended = Image.blend(original_pil, heatmap_img, alpha=alpha)
    blended.save(output_path)

def select_images(class_names):
    csv_path = os.path.join(ROOT, "plots", "test_predictions.csv")
    if ONLY_MISCLASSIFIED and os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if {"filename","true_label","pred_label"}.issubset(df.columns):
                mis = df[df["true_label"] != df["pred_label"]]
                if len(mis) > 0:
                    return [(str(r["filename"]), r["true_label"]) for _, r in mis.iterrows()]
        except Exception:
            pass
    all_files = list_all_test_images(TEST_DIR, class_names)
    if MAX_IMAGES and len(all_files) > MAX_IMAGES:
        random.shuffle(all_files)
        return all_files[:MAX_IMAGES]
    return all_files

def list_all_test_images(test_dir, class_names):
    files = []
    for cls in class_names:
        cls_dir = os.path.join(test_dir, cls)
        if not os.path.exists(cls_dir):
            continue
        for root, _, filenames in os.walk(cls_dir):
            for f in sorted(filenames):
                if f.lower().endswith(EXTS):
                    files.append((os.path.join(root,f), cls))
    return files

def main():
    random.seed(SEED); np.random.seed(SEED)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found: " + MODEL_PATH)
    print("Loading model:", MODEL_PATH)
    model = load_model(MODEL_PATH, compile=False)
    class_names = find_class_names(TEST_DIR)
    if not class_names:
        raise RuntimeError("No class folders found under Testing/")
    print("Classes:", class_names)

    in_shape = model.input_shape
    target_size = (in_shape[1] or 224, in_shape[2] or 224)
    print("Model input size:", target_size)

    last_conv = find_last_conv_layer_recursive(model)
    if last_conv is None:
        raise RuntimeError("Could not find last Conv2D layer in model.")
    print("Using conv layer:", last_conv.name)

    imgs = select_images(class_names)
    print("Images selected:", len(imgs))
    os.makedirs(OUT_DIR, exist_ok=True)

    succeeded = 0
    failed = 0
    for img_path, true_label in imgs:
        try:
            arr, pil = load_and_preprocess(img_path, target_size)
            # try eager first
            heatmap = None; preds = None
            try:
                heatmap, preds = gradcam_via_eager(model, last_conv, arr, pred_index=None)
            except Exception as e:
                # fallback to symbolic backend
                heatmap, preds = gradcam_via_symbolic(model, last_conv, arr, pred_index=None)

            pred_idx = int(np.argmax(preds[0]))
            pred_label = class_names[pred_idx]
            out_name = f"{pathlib.Path(img_path).stem}__pred-{pred_label}__true-{true_label}.png"
            out_path = os.path.join(OUT_DIR, out_name)
            overlay_heatmap_on_image(pil, heatmap, out_path)
            succeeded += 1
            if succeeded % 50 == 0:
                print("Processed:", succeeded)
        except Exception as e:
            failed += 1
            print("Failed on", img_path, ":", repr(e))
    print("Done. succeeded=", succeeded, "failed=", failed, "outputs in:", OUT_DIR)

if __name__ == "__main__":
    main()
