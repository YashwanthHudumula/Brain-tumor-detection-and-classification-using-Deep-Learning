# app.py
import os
import io
import traceback
from pathlib import Path
from PIL import Image, ImageFilter, ImageOps
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, jsonify
from werkzeug.utils import secure_filename

import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input

# -----------------------
# CONFIGURATION
# -----------------------
MODEL_PATH = r"D:\Major project\saved models\densenet201_final.h5"  # change to your path
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}
IMG_SIZE = (224, 224)        # DenseNet expected size
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
GAUSSIAN_BLUR_RADIUS = 5     # smoothing radius for CAM overlay
HEATMAP_ALPHA = 0.45         # overlay strength (0..1)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "replace-with-secure-secret"

# -----------------------
# Load classifier model
# -----------------------
model = None
try:
    print("Loading classifier model from:", MODEL_PATH)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded. Input shape:", model.input_shape)
except Exception:
    print("Failed to load classifier at startup.")
    traceback.print_exc()
    model = None

# -----------------------
# Utility helpers
# -----------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def prepare_image_for_model(pil_img: Image.Image, target_size=IMG_SIZE) -> np.ndarray:
    """Return preprocessed Numpy batch ready for DenseNet preprocess_input."""
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize(target_size, Image.BILINEAR)
    arr = np.asarray(pil_img).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)  # DenseNet preprocess (expects 0-255 input)
    return arr

def find_last_conv_layer(model):
    """
    Find the last Conv2D-like layer in `model` by scanning its layers in
    reverse. Only looks at the layers directly owned by `model` (does not
    recurse into nested sub-models) - use find_base_model_and_conv() for
    models that wrap a backbone (e.g. DenseNet201) as a single nested layer.
    Returns layer object or None.
    """
    for layer in reversed(model.layers):
        cname = layer.__class__.__name__.lower()
        if "conv" in cname and "dense" not in cname:
            # matches Conv2D, SeparableConv2D, DepthwiseConv2D, etc.
            try:
                shape = layer.output.shape
            except Exception:
                continue
            if shape is not None and len(shape) == 4:
                return layer
    return None


def find_base_model_and_conv(model):
    """
    Locate the sub-model (base_model) and the last conv layer that should be
    used for Grad-CAM.

    Many transfer-learning architectures (DenseNet201, ResNet50,
    EfficientNet, etc.) are added to an outer Sequential/Functional model as
    a single *nested* layer, e.g.:

        Input -> densenet201 (a whole Model, nested) -> GAP -> Dropout -> Dense

    In that case `model.layers` only contains the nested model as one
    opaque layer, so a naive scan for Conv2D at the top level fails (this
    was the root cause of "Grad-CAM not available"). This helper first
    checks whether any top-level layer is itself a Model with conv layers
    inside it; if so, that nested model is treated as the "base_model" and
    Grad-CAM is computed against its last conv layer. Otherwise it falls
    back to looking for conv layers directly in the top-level model.

    Returns (base_model, last_conv_layer) or (None, None) if nothing found.
    """
    # 1) Look for a nested sub-model (backbone) containing conv layers.
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            conv = find_last_conv_layer(layer)
            if conv is not None:
                return layer, conv

    # 2) Fall back to conv layers living directly on the top-level model.
    conv = find_last_conv_layer(model)
    if conv is not None:
        return model, conv

    return None, None


def make_gradcam_overlay(model, img_pil: Image.Image, class_idx=None,
                         base_model=None, last_conv_layer=None,
                         blur_radius=GAUSSIAN_BLUR_RADIUS,
                         alpha=HEATMAP_ALPHA):
    """
    Returns (overlay_pil, heatmap_pil)
    overlay_pil: original image with a colored translucent heatmap overlay (RGBA)
    heatmap_pil: grayscale heatmap (L) normalized 0..255 (smoothed)

    Implementation follows the standard two-stage Grad-CAM recipe (as in
    the official Keras Grad-CAM guide) so it works correctly even when the
    convolutional backbone is a *nested* sub-model rather than living
    directly on `model`:

      1. `last_conv_layer_model`: maps the raw image -> feature maps of the
         last conv layer (inside the backbone, if nested).
      2. `classifier_model`: maps those feature maps -> the model's final
         prediction, by replaying every layer that comes *after* the
         backbone/conv layer.

    A `tf.GradientTape` is used to get the gradient of the predicted
    class's score with respect to the conv feature maps, which are then
    combined into the Grad-CAM heatmap.
    """
    if model is None:
        raise RuntimeError("Model not loaded.")

    # Resolve which (sub-)model and conv layer to use.
    if base_model is None or last_conv_layer is None:
        base_model, last_conv_layer = find_base_model_and_conv(model)
        if base_model is None or last_conv_layer is None:
            raise RuntimeError("Could not find a Conv2D layer in the model for Grad-CAM.")

    # Prepare input for model using same preprocessing as the classifier.
    img_resized = img_pil.copy().resize(IMG_SIZE, Image.BILINEAR)
    arr = prepare_image_for_model(img_resized, IMG_SIZE)  # shape (1,H,W,3)
    arr_tensor = tf.convert_to_tensor(arr)

    # --- Stage 1: input -> last conv feature maps -----------------------
    last_conv_layer_model = tf.keras.models.Model(
        inputs=base_model.input, outputs=last_conv_layer.output
    )

    # --- Stage 2: conv feature maps -> final prediction ------------------
    if base_model is model:
        # Conv layer lives directly on the top-level model; replay every
        # layer that comes after it.
        layer_names = [l.name for l in model.layers]
        conv_idx = layer_names.index(last_conv_layer.name)
        remaining_layers = model.layers[conv_idx + 1:]
    else:
        # Conv layer lives inside a nested backbone; replay the rest of the
        # backbone (if any) followed by everything after the backbone in
        # the outer model.
        base_layer_names = [l.name for l in base_model.layers]
        conv_idx = base_layer_names.index(last_conv_layer.name)
        remaining_layers = list(base_model.layers[conv_idx + 1:])

        outer_layer_names = [l.name for l in model.layers]
        base_idx = outer_layer_names.index(base_model.name)
        remaining_layers += list(model.layers[base_idx + 1:])

    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer in remaining_layers:
        x = layer(x)
    classifier_model = tf.keras.models.Model(classifier_input, x)

    # --- Compute Grad-CAM --------------------------------------------------
    with tf.GradientTape() as tape:
        conv_output = last_conv_layer_model(arr_tensor)
        tape.watch(conv_output)
        predictions = classifier_model(conv_output)
        if class_idx is None:
            class_idx = int(tf.argmax(predictions[0]))
        top_class_channel = predictions[:, class_idx]

    grads = tape.gradient(top_class_channel, conv_output)
    if grads is None:
        raise RuntimeError("Gradients returned None during Grad-CAM computation.")

    # Global-average-pool the gradients over the spatial dimensions.
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # shape (channels,)

    conv_output = conv_output[0]  # (H, W, C)
    cam = conv_output @ pooled_grads[..., tf.newaxis]
    cam = tf.squeeze(cam)  # (H, W)

    # ReLU and normalize to 0..1
    cam = tf.nn.relu(cam)
    cam = cam.numpy()
    if np.max(cam) > 0:
        cam = cam / (np.max(cam) + 1e-8)
    else:
        cam = np.zeros_like(cam)

    # Resize CAM to original image size
    cam_tf = tf.convert_to_tensor(cam[..., np.newaxis][np.newaxis, ...], dtype=tf.float32)  # (1,H,W,1)
    target_h, target_w = img_pil.size[1], img_pil.size[0]
    cam_resized = tf.image.resize(cam_tf, size=(target_h, target_w), method="bilinear")
    cam_resized = tf.squeeze(cam_resized).numpy()  # (H_orig, W_orig), values 0..1

    # Smooth the heatmap with PIL GaussianBlur to reduce pixelation
    heatmap_img = Image.fromarray(np.uint8(np.clip(cam_resized * 255.0, 0, 255))).convert("L")
    if blur_radius > 0:
        heatmap_img = heatmap_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    heat_arr = np.asarray(heatmap_img).astype("float32") / 255.0  # 0..1

    # Colorize the heatmap with a "jet"-like colormap for a proper Grad-CAM
    # look (blue -> green -> yellow -> red) instead of a flat red overlay.
    colored = _apply_jet_colormap(heat_arr)  # (H, W, 3) uint8

    orig_rgba = img_pil.convert("RGBA")
    color_rgba = Image.fromarray(colored, mode="RGB").convert("RGBA")
    alpha_channel = (heat_arr * (alpha * 255.0)).astype(np.uint8)
    color_rgba.putalpha(Image.fromarray(alpha_channel, mode="L"))

    # Composite colored heatmap over original image
    composed = Image.alpha_composite(orig_rgba, color_rgba)

    return composed, heatmap_img


def _apply_jet_colormap(heat_arr: np.ndarray) -> np.ndarray:
    """
    Map a single-channel array of values in [0, 1] to an RGB "jet"-style
    colormap without requiring matplotlib. Returns a uint8 (H, W, 3) array.
    """
    h = np.clip(heat_arr, 0.0, 1.0)

    r = np.clip(1.5 - np.abs(4 * h - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * h - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * h - 1), 0, 1)

    rgb = np.stack([r, g, b], axis=-1)
    return np.uint8(np.clip(rgb * 255.0, 0, 255))

# -----------------------
# Routes
# -----------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=False)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        flash("Model not loaded on server. Check logs.")
        return redirect(url_for("index"))

    if "file" not in request.files:
        flash("No file uploaded.")
        return redirect(url_for("index"))
    f = request.files["file"]
    if f.filename == "":
        flash("No file selected.")
        return redirect(url_for("index"))
    if not allowed_file(f.filename):
        flash("File type not allowed.")
        return redirect(url_for("index"))

    filename = secure_filename(f.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    try:
        f.save(save_path)
    except Exception as e:
        flash(f"Failed to save file: {e}")
        return redirect(url_for("index"))

    # Open image
    try:
        orig = Image.open(save_path).convert("RGB")
    except Exception as e:
        flash(f"Cannot open image: {e}")
        return redirect(url_for("index"))

    # Classification
    try:
        arr = prepare_image_for_model(orig, IMG_SIZE)
        preds = model.predict(arr, verbose=0)[0]
    except Exception as e:
        traceback.print_exc()
        flash(f"Prediction failed: {e}")
        return redirect(url_for("index"))

    top_idx = int(np.argmax(preds))
    top_prob = float(preds[top_idx])
    pred_label = CLASS_NAMES[top_idx] if top_idx < len(CLASS_NAMES) else str(top_idx)
    prob_map = {cls: float(preds[i]) for i, cls in enumerate(CLASS_NAMES)}

    if pred_label == "notumor":
        detection = {"status": "No tumor", "explanation": "Classifier predicted 'notumor' — no tumor detected."}
    else:
        detection = {"status": "Tumor found", "explanation": f"Classifier predicted '{pred_label}' — tumor likely present."}

    # Build Grad-CAM and save overlay into 'segmentation' area
    gradcam_file = None
    heatmap_file = None
    gradcam_error = None
    try:
        base_model, last_conv = find_base_model_and_conv(model)
        if base_model is None or last_conv is None:
            raise RuntimeError("No Conv2D layer found in model (or any nested backbone) for Grad-CAM.")
        overlay_pil, heatmap_pil = make_gradcam_overlay(
            model, orig, class_idx=top_idx, base_model=base_model, last_conv_layer=last_conv
        )
        base, ext = os.path.splitext(filename)
        gradcam_file = f"{base}_gradcam.png"
        heatmap_file = f"{base}_heatmap.png"
        overlay_pil.convert("RGBA").save(os.path.join(app.config["UPLOAD_FOLDER"], gradcam_file))
        heatmap_pil.save(os.path.join(app.config["UPLOAD_FOLDER"], heatmap_file))

    except Exception as e:
        # If gradcam fails, log but continue; segmentation area will show message
        traceback.print_exc()
        print("Grad-CAM generation failed:", e)
        gradcam_file = None
        heatmap_file = None
        gradcam_error = str(e)

    # Render result template; segmentation area will show gradcam overlay if present
    return render_template(
        "result.html",
        filename=filename,
        detection=detection,
        segmentation={
            "status": "Grad-CAM",
            "overlay_file": gradcam_file,
            "heatmap_file": heatmap_file,
            "explanation": gradcam_error,
        },
        pred_label=pred_label,
        top_prob=f"{top_prob*100:.2f}%",
        prob_map=prob_map
    )


@app.route("/predict_json", methods=["POST"])
def predict_json():
    if model is None:
        return jsonify({"error": "Model not loaded."}), 500

    # Accept file field
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400
    f = request.files["file"]
    try:
        img = Image.open(io.BytesIO(f.read())).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Failed to open image: {e}"}), 400

    try:
        arr = prepare_image_for_model(img, IMG_SIZE)
        preds = model.predict(arr, verbose=0)[0]
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    top_idx = int(np.argmax(preds))
    top_prob = float(preds[top_idx])
    pred_label = CLASS_NAMES[top_idx] if top_idx < len(CLASS_NAMES) else str(top_idx)
    prob_map = {cls: float(preds[i]) for i, cls in enumerate(CLASS_NAMES)}

    # Grad-CAM as base64 or saved file? we'll save file to uploads and return filename.
    gradcam_name = None
    try:
        base_model, last_conv = find_base_model_and_conv(model)
        if base_model is not None and last_conv is not None:
            overlay_pil, heatmap_pil = make_gradcam_overlay(
                model, img, class_idx=top_idx, base_model=base_model, last_conv_layer=last_conv
            )
            gradcam_name = "pred_json_gradcam.png"
            heatmap_name = "pred_json_heatmap.png"
            overlay_pil.convert("RGBA").save(os.path.join(app.config["UPLOAD_FOLDER"], gradcam_name))
            heatmap_pil.save(os.path.join(app.config["UPLOAD_FOLDER"], heatmap_name))
    except Exception:
        traceback.print_exc()
        gradcam_name = None
        heatmap_name = None

    detection = {"status": "No tumor" if pred_label == "notumor" else "Tumor found"}

    return jsonify({
        "detection": detection,
        "classification": {"pred_idx": top_idx, "pred_label": pred_label, "prob": top_prob, "probs": prob_map},
        "gradcam_file": gradcam_name if gradcam_name else None
    })

# -----------------------
# Run dev server
# -----------------------
if __name__ == "__main__":
    # For development only. Use gunicorn/waitress behind nginx in production.
    app.run(host="0.0.0.0", port=5000, debug=True)
