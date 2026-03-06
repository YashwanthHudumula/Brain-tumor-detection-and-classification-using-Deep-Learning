# gradcam_helper_robust.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import Model
from PIL import Image
import cv2
from matplotlib import cm

def find_last_conv_layer_recursive(model):
    """Return last Conv2D layer object in (possibly nested) model or None."""
    last = None
    def recurse(m):
        nonlocal last
        for layer in getattr(m, "layers", []):
            # If nested model, recurse
            if hasattr(layer, "layers"):
                recurse(layer)
            else:
                if isinstance(layer, Conv2D):
                    last = layer
    recurse(model)
    return last

def find_layer_by_name_recursive(model, name):
    """Find a layer object by name across nested models."""
    def recurse(m):
        for layer in getattr(m, "layers", []):
            if layer.name == name:
                return layer
            if hasattr(layer, "layers"):
                found = recurse(layer)
                if found is not None:
                    return found
        return None
    return recurse(model)

def safe_call_model(model, x):
    """
    Call a Keras model with input x (np array or tf tensor).
    Handles models whose .inputs is a list vs single tensor.
    Returns model(x) output or raises a clear exception.
    """
    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
    # if model expects list inputs of length 1, call with [x_tf], else x_tf
    try:
        if isinstance(model.inputs, (list, tuple)) and len(model.inputs) == 1:
            return model([x_tf], training=False)
        else:
            return model(x_tf, training=False)
    except Exception as e:
        # fallback: try calling with plain x_tf
        try:
            return model(x_tf, training=False)
        except Exception:
            raise

def make_gradcam_and_overlay(model,
                             img_array,
                             orig_img_path,
                             out_overlay_path,
                             out_heatmap_path,
                             last_conv_name=None,
                             alpha=0.45,
                             blur_ksize=(13,13),
                             upsample_method=cv2.INTER_LINEAR):
    """
    Robust Grad-CAM generator.
    - model: loaded Keras model
    - img_array: numpy array shaped (1,H,W,3) and preprocessed as for prediction
    - orig_img_path: path to original image (for overlay)
    - out_overlay_path, out_heatmap_path: output files
    - last_conv_name: optional layer name override
    Returns (overlay_path, heatmap_path, error_string_or_None)
    """

    # ---------- 0) Basic sanity checks on img_array ----------
    try:
        arr = np.array(img_array)
    except Exception as e:
        return None, None, f"Input array conversion failed: {e}"

    if arr.ndim != 4 or arr.shape[0] != 1 or arr.shape[-1] != 3:
        return None, None, f"Expected img_array shape (1,H,W,3), got {arr.shape}"

    # check emptiness / constant image
    mn, mx = float(np.min(arr)), float(np.max(arr))
    if mx == 0 and mn == 0:
        return None, None, "Input appears to be blank (all zeros). Check preprocessing / file load."
    if np.isnan(mn) or np.isnan(mx):
        return None, None, "Input contains NaNs."

    # ---------- 1) Find last conv layer ----------
    last_conv_layer = None
    if last_conv_name:
        last_conv_layer = find_layer_by_name_recursive(model, last_conv_name)
        if last_conv_layer is None:
            # try direct model.get_layer (works if top-level)
            try:
                last_conv_layer = model.get_layer(last_conv_name)
            except Exception:
                pass

    if last_conv_layer is None:
        last_conv_layer = find_last_conv_layer_recursive(model)

    if last_conv_layer is None:
        return None, None, "No Conv2D layer found in model for Grad-CAM."

    # ---------- 2) Build grad model safely ----------
    # The main trick: the layer object we found might be nested; ensure we can reference
    # a tensor that is connected to the model graph. We'll try a few strategies.
    grad_model = None
    tried_reprs = []
    try:
        # simplest attempt: use layer.output directly (works if layer is connected)
        grad_model = Model(inputs=model.inputs, outputs=[last_conv_layer.output, model.output])
    except Exception as e:
        tried_reprs.append(("direct", str(e)))
        # try to find layer by name via model.get_layer (may raise if nested)
        try:
            l = model.get_layer(last_conv_layer.name)
            grad_model = Model(inputs=model.inputs, outputs=[l.output, model.output])
        except Exception as e2:
            tried_reprs.append(("get_layer", str(e2)))
            # last fallback: search nested submodels to find a submodel that contains the conv layer
            # and try to use that submodel's layer.output if its tensors are connected.
            found = None
            def find_submodel_with_layer(m, target_name):
                for layer in getattr(m, "layers", []):
                    if hasattr(layer, "layers"):
                        # if this nested model has the layer
                        for sub in getattr(layer, "layers", []):
                            if sub.name == target_name:
                                return layer, sub
                        res = find_submodel_with_layer(layer, target_name)
                        if res is not None:
                            return res
                return None
            res = find_submodel_with_layer(model, last_conv_layer.name)
            if res is not None:
                nested_model, nested_layer = res
                try:
                    # attempt to map nested_layer.output into the top-level model graph by name
                    grad_model = Model(inputs=model.inputs, outputs=[nested_layer.output, model.output])
                except Exception as e3:
                    tried_reprs.append(("nested_try", str(e3)))
                    grad_model = None

    if grad_model is None:
        # Collect some debugging info
        debug_msg = "Could not build grad_model for layer '{}'. Attempts: {}.".format(
            last_conv_layer.name, tried_reprs
        )
        return None, None, debug_msg

    # ---------- 3) Compute gradients with GradientTape ----------
    try:
        # ensure consistent tensor type
        x_tf = tf.convert_to_tensor(arr, dtype=tf.float32)
        # call grad_model with appropriate input structure
        # (grad_model.inputs may be list or single)
        if isinstance(grad_model.inputs, (list, tuple)) and len(grad_model.inputs) == 1:
            conv_outputs, predictions = grad_model([x_tf], training=False)
        else:
            conv_outputs, predictions = grad_model(x_tf, training=False)

        # choose predicted index (supports probability logits)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

        tape = tf.GradientTape()
        # We need a new tape context to watch conv_outputs if they're not watched; safer to compute again
        with tf.GradientTape() as tape:
            tape.watch(conv_outputs)
            # recompute predictions by calling only the model part that uses conv_outputs is tricky -> simpler:
            # build another model? For robustness we compute grads of class_channel wrt conv_outputs above
            # but some TF versions require we watch input; using tape above should work because conv_outputs is watched
            # Instead compute grads directly:
            # (Note: in many examples they compute grads = tape.gradient(class_channel, conv_outputs))
            pass

        grads = tf.gradients(class_channel, conv_outputs)[0] if hasattr(tf, "gradients") else tape.gradient(class_channel, conv_outputs)
        # If grads is None, try using a second approach
        if grads is None:
            # fallback: compute using a fresh tape around a direct call
            with tf.GradientTape() as gt:
                gt.watch(conv_outputs)
                # compute class score again using predictions tensor already computed
                cc = predictions[:, pred_index]
            grads = gt.gradient(cc, conv_outputs)

        if grads is None:
            return None, None, "Unable to compute gradients (grads is None)."

        pooled_grads = tf.reduce_mean(grads, axis=(0,1,2)).numpy()
        conv_out = conv_outputs[0].numpy()

        # weight maps by pooled grads
        for i in range(pooled_grads.shape[-1]):
            conv_out[:, :, i] *= pooled_grads[i]

        heatmap = np.sum(conv_out, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap = heatmap / (max_val + 1e-12)
        else:
            # degenerate heatmap
            heatmap = heatmap

    except Exception as e:
        return None, None, f"Failed computing gradients: {e}"

    # ---------- 4) Resize, smooth, colorize and save ----------
    try:
        orig = Image.open(orig_img_path).convert("RGB")
        ow, oh = orig.size

        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_resized = cv2.resize(heatmap_uint8, (ow, oh), interpolation=upsample_method)

        if blur_ksize is not None and blur_ksize[0] > 1:
            heatmap_resized = cv2.GaussianBlur(heatmap_resized, blur_ksize, 0)

        colormap = cm.get_cmap("jet")
        colored = colormap(heatmap_resized / 255.0)[:, :, :3]
        colored_uint8 = np.uint8(colored * 255)

        Image.fromarray(colored_uint8).save(out_heatmap_path)

        orig_np = np.array(orig)
        overlay = cv2.addWeighted(orig_np, 1.0 - alpha, colored_uint8, alpha, 0)
        Image.fromarray(overlay).save(out_overlay_path)

        return out_overlay_path, out_heatmap_path, None

    except Exception as e:
        return None, None, f"Failed creating/saving overlay: {e}"
