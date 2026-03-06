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
    Find the last Conv2D layer in the model by scanning reversed layers.
    Returns layer object or None.
    """
    for layer in reversed(model.layers):
        # Some wrappers (Functional model as nested) - try to detect conv by name and ndims
        if hasattr(layer, "output_shape"):
            out_shape = layer.output_shape
            # out_shape may be tuple or list; look for 4D tensors (batch,H,W,C)
            if out_shape is None:
                continue
            if isinstance(out_shape, (list, tuple)) and len(out_shape) == 4:
                # check layer type name contains 'conv' (robust) or class name contains 'Conv'
                lname = layer.name.lower()
                cname = layer.__class__.__name__.lower()
                if "conv" in lname or "conv" in cname:
                    return layer
    return None

def make_gradcam_overlay(model, img_pil: Image.Image, class_idx=None,
                         last_conv_layer=None, blur_radius=GAUSSIAN_BLUR_RADIUS,
                         alpha=HEATMAP_ALPHA):
    """
    Returns (overlay_pil, heatmap_pil)
    overlay_pil: original image with translucent heatmap overlay (RGBA)
    heatmap_pil: grayscale heatmap (L) normalized 0..255 (smoothed)
    """
    if model is None:
        raise RuntimeError("Model not loaded.")

    # Ensure last_conv_layer object
    if last_conv_layer is None:
        last_conv_layer = find_last_conv_layer(model)
        if last_conv_layer is None:
            raise RuntimeError("Could not find a Conv2D layer in the model for Grad-CAM.")

    # Prepare input for model using same preprocessing
    model_input_size = IMG_SIZE
    img_for_model = img_pil.copy()
    img_resized = img_for_model.resize(model_input_size, Image.BILINEAR)
    arr = prepare_image_for_model(img_resized, model_input_size)  # shape (1,H,W,3)

    # Build grad model that outputs conv features + predictions
    # Some models may have nested models; use model.inputs and last_conv_layer.output
    try:
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, model.output]
        )
    except Exception:
        # attempt to access nested model if top-level layer is a nested model (e.g. named densenet201)
        # find a layer that is a Functional model with name containing densenet and use its layers
        nested = None
        for layer in model.layers:
            if hasattr(layer, "layers") and getattr(layer, "name", "").lower().startswith("densenet"):
                nested = layer
                break
        if nested is not None:
            last_conv_layer = find_last_conv_layer(nested)
            if last_conv_layer is None:
                raise RuntimeError("Couldn't find last conv in nested model.")
            grad_model = tf.keras.models.Model(inputs=model.inputs,
                                               outputs=[last_conv_layer.output, model.output])
        else:
            raise

    # Predict class if not provided
    if class_idx is None:
        preds = model.predict(arr, verbose=0)
        class_idx = int(np.argmax(preds[0]))

    # Compute gradients
    arr_tensor = tf.convert_to_tensor(arr)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(arr_tensor)
        # If predictions shape (1, num_classes)
        top_class_channel = predictions[:, class_idx]

    # Gradient of the output neuron (for class_idx) wrt conv outputs
    grads = tape.gradient(top_class_channel, conv_outputs)
    if grads is None:
        raise RuntimeError("Gradients returned None during Grad-CAM computation.")

    # Compute channel-wise mean of gradients (global average pooling)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # shape (channels,)

    # Multiply each channel in feature map array by corresponding weight
    conv_outputs = conv_outputs[0]  # (H, W, C)
    pooled_grads = pooled_grads[..., tf.newaxis, tf.newaxis] if len(pooled_grads.shape)==1 else pooled_grads
    # compute weighted combination
    weighted = conv_outputs * pooled_grads  # broadcasting
    cam = tf.reduce_sum(weighted, axis=-1)  # (H, W)

    # ReLU and normalize
    cam = tf.nn.relu(cam)
    cam = cam.numpy()
    if np.max(cam) > 0:
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
    else:
        cam = np.zeros_like(cam)

    # Resize CAM to original image size
    cam_tf = tf.convert_to_tensor(cam[..., np.newaxis][np.newaxis, ...], dtype=tf.float32)  # (1,H,W,1)
    target_size = img_pil.size[::-1]  # tensorflow expects (height, width)
    cam_resized = tf.image.resize(cam_tf, size=(target_size[0], target_size[1]), method="bilinear")
    cam_resized = tf.squeeze(cam_resized).numpy()  # (H_orig, W_orig), values 0..1

    # Smooth the heatmap with PIL GaussianBlur to reduce pixelation
    heatmap_img = Image.fromarray(np.uint8(np.clip(cam_resized * 255.0, 0, 255))).convert("L")
    if blur_radius > 0:
        heatmap_img = heatmap_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Build red overlay: use heatmap as alpha for red color
    heat_arr = np.asarray(heatmap_img).astype("float32") / 255.0  # 0..1
    # Create color overlay (RGBA) where red channel = heat, alpha = heat * alpha
    orig_rgba = img_pil.convert("RGBA")
    overlay = Image.new("RGBA", orig_rgba.size, (255, 0, 0, 0))
    alpha_channel = (heat_arr * (alpha * 255.0)).astype(np.uint8)
    overlay.putalpha(Image.fromarray(alpha_channel, mode="L"))

    # Composite overlay over original
    composed = Image.alpha_composite(orig_rgba, overlay)

    return composed, heatmap_img

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
    try:
        last_conv = find_last_conv_layer(model)
        if last_conv is None:
            raise RuntimeError("No Conv2D layer found in model for Grad-CAM.")
        overlay_pil, heatmap_pil = make_gradcam_overlay(model, orig, class_idx=top_idx, last_conv_layer=last_conv)
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

    # Render result template; segmentation area will show gradcam overlay if present
    return render_template(
        "result.html",
        filename=filename,
        detection=detection,
        segmentation={"status": "Grad-CAM", "overlay_file": gradcam_file, "heatmap_file": heatmap_file},
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
        last_conv = find_last_conv_layer(model)
        if last_conv is not None:
            overlay_pil, heatmap_pil = make_gradcam_overlay(model, img, class_idx=top_idx, last_conv_layer=last_conv)
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
