# repair_and_resave_v2.py
import os
import traceback
from pathlib import Path
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3, InceptionV3, DenseNet201

# --- EDIT PATHS if needed ---
SAVED_DIR = r"D:\Major project\saved models"
EN_FILE = os.path.join(SAVED_DIR, "best_model_20251117-231323.h5")   # EfficientNet-B3 file
IN_FILE = os.path.join(SAVED_DIR, "brain_tumor_inceptionv3.h5")     # Inception file
DN_FILE = os.path.join(SAVED_DIR, "densenet201_final.h5")           # DenseNet (already good)

OUT_DIR = r"D:\Major project\fixed_models_v2"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_CLASSES = 4

def list_h5_model_groups(h5path):
    with h5py.File(h5path, "r") as f:
        groups = []
        if "model_weights" in f:
            groups = list(f["model_weights"].keys())
        return groups

def build_efficientnetb3_model(input_shape=(300,300,3), num_classes=NUM_CLASSES):
    base = EfficientNetB3(include_top=False, weights=None, input_shape=input_shape)
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = models.Model(base.input, outputs, name="efficientnetb3_custom")
    return model

def build_inceptionv3_model(input_shape=(299,299,3), num_classes=NUM_CLASSES):
    base = InceptionV3(include_top=False, weights=None, input_shape=input_shape)
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = models.Model(base.input, outputs, name="inceptionv3_custom")
    return model

def load_by_name_with_report(model, h5path):
    print(f"Loading weights by_name from: {h5path}")
    available_groups = []
    try:
        with h5py.File(h5path, "r") as f:
            if "model_weights" in f:
                available_groups = list(f["model_weights"].keys())
        print("Top-level layer groups in file (sample):", available_groups[:30])
    except Exception as e:
        print("Could not open h5 file to list groups:", e)

    # Attempt load_weights
    try:
        model.load_weights(h5path, by_name=True, skip_mismatch=True)
        print("load_weights(by_name=True, skip_mismatch=True) completed without exception.")
    except Exception as e:
        print("load_weights threw exception:")
        traceback.print_exc()
        return False, available_groups

    # Now produce a matching report: which model layer names are present in file
    model_layer_names = [layer.name for layer in model.layers]
    in_file = set(available_groups)
    matched = [n for n in model_layer_names if n in in_file]
    missing_in_file = [n for n in model_layer_names if n not in in_file and (len(model.get_layer(n).weights) > 0)]
    extra_in_file = [n for n in in_file if n not in model_layer_names]

    print(f"Number of model layers: {len(model_layer_names)}")
    print(f"Layers present in file and in model (matched): {len(matched)} (sample 30): {matched[:30]}")
    print(f"Layers in model but NOT in file (and have weights) count: {len(missing_in_file)} (sample 30): {missing_in_file[:30]}")
    print(f"Extra layer groups in file not found in current model count: {len(extra_in_file)} (sample 30): {extra_in_file[:30]}")

    return True, available_groups

def sanity_predict_and_save(model, out_base):
    try:
        # Sanity run
        size = model.input_shape[1:3]
        dummy = np.zeros((1, size[0], size[1], 3), dtype=np.float32)
        _ = model.predict(dummy, verbose=0)
        print("Sanity forward pass OK for input size", size)
    except Exception as e:
        print("Sanity forward pass FAILED:", e)
        traceback.print_exc()
        return False

    # Save SavedModel directory
    savedmodel_dir = out_base + "_savedmodel"
    try:
        model.save(savedmodel_dir, include_optimizer=False)
        print("Saved SavedModel to:", savedmodel_dir)
    except Exception as e:
        print("Failed saving SavedModel:", e)
        traceback.print_exc()

    # Also save HDF5 (optional)
    h5_out = out_base + ".h5"
    try:
        model.save(h5_out, include_optimizer=False)
        print("Also saved HDF5 to:", h5_out)
    except Exception as e:
        print("Failed saving HDF5:", e)
        traceback.print_exc()

    return True

def repair_efficientnetb3():
    print("\n--- Repair EfficientNet-B3 ---")
    print("File groups in EN_FILE:")
    print(list_h5_model_groups(EN_FILE))
    model = build_efficientnetb3_model()
    ok, groups = load_by_name_with_report(model, EN_FILE)
    if not ok:
        print("Loading failed for EfficientNetB3.")
        return False
    out_base = os.path.join(OUT_DIR, "efficientnetb3_fixed")
    return sanity_predict_and_save(model, out_base)

def repair_inceptionv3():
    print("\n--- Repair InceptionV3 ---")
    print("File groups in IN_FILE:")
    print(list_h5_model_groups(IN_FILE))
    model = build_inceptionv3_model()
    ok, groups = load_by_name_with_report(model, IN_FILE)
    if not ok:
        print("Loading failed for InceptionV3.")
        return False
    out_base = os.path.join(OUT_DIR, "inceptionv3_fixed")
    return sanity_predict_and_save(model, out_base)

if __name__ == "__main__":
    print("TensorFlow version:", tf.__version__)
    print("H5 files to repair:")
    print(" EfficientNet:", EN_FILE)
    print(" Inception :", IN_FILE)
    print("Output folder:", OUT_DIR)
    print("NUM_CLASSES:", NUM_CLASSES)

    res_en = repair_efficientnetb3()
    print("EfficientNetB3 repair result:", res_en)
    res_in = repair_inceptionv3()
    print("InceptionV3 repair result:", res_in)
    print("Done.")
