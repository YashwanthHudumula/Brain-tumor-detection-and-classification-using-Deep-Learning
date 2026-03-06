# train_inceptionv3.py
import os, pathlib, shutil, argparse, json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import csv

IMG_EXTS = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}

def ensure_classes_ok(root, class_names):
    root = pathlib.Path(root)
    missing = [c for c in class_names if not (root / c).exists()]
    if missing:
        raise ValueError(f"Missing class folders in {root}: {missing}")

def stratified_split(source_dir, out_root, class_names, val_split=0.15, test_split=0.15, seed=42):
    source_dir = pathlib.Path(source_dir)
    out_root = pathlib.Path(out_root)
    train_dir = out_root / 'train'
    val_dir = out_root / 'val'
    test_dir = out_root / 'test'
    for d in [train_dir, val_dir, test_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
        for c in class_names:
            (d / c).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    for c in class_names:
        files = []
        for ext in ['*.jpg','*.jpeg','*.png','*.bmp','*.tif','*.tiff']:
            files.extend(list((source_dir / c).glob(ext)))
        files = np.array(files)
        if len(files) == 0:
            raise ValueError(f"No images found for class '{c}' in {source_dir/c}")

        rng.shuffle(files)
        n = len(files)
        n_test = max(1, int(n * test_split))
        n_val  = max(1, int(n * val_split))
        n_train = max(1, n - n_val - n_test)

        splits = {
            train_dir / c: files[:n_train],
            val_dir / c: files[n_train:n_train+n_val],
            test_dir / c: files[n_train+n_val:]
        }
        for dst, subset in splits.items():
            for f in subset:
                shutil.copy2(f, dst)

    return str(train_dir), str(val_dir), str(test_dir)

def build_model(img_size, num_classes):
    preprocess = tf.keras.applications.inception_v3.preprocess_input
    base = tf.keras.applications.InceptionV3(
        include_top=False, weights='imagenet', input_shape=img_size + (3,)
    )
    base.trainable = False

    aug = keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.12),
        layers.RandomTranslation(0.06, 0.06),
        layers.RandomContrast(0.12),
    ], name="augmentation")

    inputs = keras.Input(shape=img_size + (3,))
    x = aug(inputs)
    x = preprocess(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax',
                           kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    model = keras.Model(inputs, outputs)
    return model, base, preprocess

def plot_and_save_history(history, out_folder):
    os.makedirs(out_folder, exist_ok=True)
    history_json = history.history
    with open(os.path.join(out_folder, 'history.json'), 'w', encoding='utf-8') as f:
        json.dump(history_json, f, indent=2)

    # Plot accuracy
    plt.figure()
    plt.plot(history_json.get('accuracy', []), label='train_acc')
    plt.plot(history_json.get('val_accuracy', []), label='val_acc')
    plt.title('Train vs Val Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_folder, 'train_val_accuracy.png'))
    plt.close()

    # Plot loss
    plt.figure()
    plt.plot(history_json.get('loss', []), label='train_loss')
    plt.plot(history_json.get('val_loss', []), label='val_loss')
    plt.title('Train vs Val Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_folder, 'train_val_loss.png'))
    plt.close()

def save_reports(y_true, y_pred, class_names, out_folder):
    os.makedirs(out_folder, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted label'); plt.ylabel('True label'); plt.title('Confusion Matrix')
    plt.savefig(os.path.join(out_folder, 'confusion_matrix.png'))
    plt.close()

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    # TXT
    with open(os.path.join(out_folder, 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(classification_report(y_true, y_pred, target_names=class_names))
    # JSON
    with open(os.path.join(out_folder, 'classification_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    # per-class CSV
    csv_path = os.path.join(out_folder, 'per_class_metrics.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['class','precision','recall','f1-score','support'])
        for c in class_names:
            r = report.get(c, {})
            writer.writerow([c, r.get('precision',0), r.get('recall',0), r.get('f1-score',0), r.get('support',0)])

    # overall metrics bar
    overall = {
        "Accuracy": float(np.mean(np.array(y_true) == np.array(y_pred))),
        "Precision": float(np.mean([report[c]['precision'] for c in class_names])),
        "Recall": float(np.mean([report[c]['recall'] for c in class_names])),
        "F1-Score": float(np.mean([report[c]['f1-score'] for c in class_names]))
    }
    plt.figure(figsize=(6,4))
    names = list(overall.keys()); vals = list(overall.values())
    plt.bar(names, vals)
    plt.ylim(0,1)
    plt.title("Overall metrics")
    plt.savefig(os.path.join(out_folder, 'metrics_bar.png'))
    plt.close()
    return overall

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--source_data_dir', default='raw_data')
    parser.add_argument('--do_split', action='store_true')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, nargs=2, default=[299,299])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_split', type=float, default=0.15)
    parser.add_argument('--test_split', type=float, default=0.15)
    parser.add_argument('--classes', type=str, nargs='+', default=['glioma','meningioma','pituitary','notumor'])
    args = parser.parse_args()

    IMG_SIZE = tuple(args.img_size)
    CLASS_NAMES = args.classes

    os.makedirs(args.data_root, exist_ok=True)
    if args.do_split:
        ensure_classes_ok(args.source_data_dir, CLASS_NAMES)
        train_dir, val_dir, test_dir = stratified_split(args.source_data_dir, args.data_root, CLASS_NAMES, args.val_split, args.test_split, args.seed)
    else:
        train_dir = os.path.join(args.data_root, 'train')
        val_dir = os.path.join(args.data_root, 'val')
        test_dir = os.path.join(args.data_root, 'test')

    print('Train dir:', train_dir)
    print('Val dir  :', val_dir)
    print('Test dir :', test_dir)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(train_dir, labels='inferred', label_mode='int', class_names=CLASS_NAMES, seed=args.seed, image_size=IMG_SIZE, batch_size=args.batch_size, shuffle=True)
    val_ds   = tf.keras.preprocessing.image_dataset_from_directory(val_dir, labels='inferred', label_mode='int', class_names=CLASS_NAMES, seed=args.seed, image_size=IMG_SIZE, batch_size=args.batch_size, shuffle=False)
    test_ds  = tf.keras.preprocessing.image_dataset_from_directory(test_dir, labels='inferred', label_mode='int', class_names=CLASS_NAMES, seed=args.seed, image_size=IMG_SIZE, batch_size=args.batch_size, shuffle=False)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds   = val_ds.prefetch(AUTOTUNE)
    test_ds  = test_ds.prefetch(AUTOTUNE)

    model, base, preprocess = build_model(IMG_SIZE, len(CLASS_NAMES))

    all_labels = [int(y.numpy()) for _, y in train_ds.unbatch().take(100000)]
    classes = np.arange(len(CLASS_NAMES))
    weights = compute_class_weight('balanced', classes=classes, y=np.array(all_labels))
    class_weight = {i: float(w) for i, w in enumerate(weights)}
    print('Class weights:', class_weight)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    ckpt_path = 'best_inceptionv3.keras'
    callbacks = [
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_accuracy', save_best_only=True, mode='max'),
        keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True, monitor='val_accuracy'),
        keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3, monitor='val_loss')
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, class_weight=class_weight, verbose=1, callbacks=callbacks)

    # fine-tune: unfreeze some base layers
    fine_tune_from = 180
    for i, layer in enumerate(base.layers):
        layer.trainable = (i >= fine_tune_from)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history_ft = model.fit(train_ds, validation_data=val_ds, epochs=max(10, args.epochs//2), class_weight=class_weight, verbose=1, callbacks=callbacks)

    # combine histories for plotting (append)
    # use history.history keys
    hist_combined = {}
    for k,v in history.history.items():
        hist_combined[k] = v
    for k,v in history_ft.history.items():
        hist_combined.setdefault(k, []).extend(v)

    # evaluate on test set
    y_true = []
    y_pred = []
    for x_batch, y_batch in test_ds:
        preds = model.predict(x_batch)
        y_true.extend(y_batch.numpy().tolist())
        y_pred.extend(np.argmax(preds, axis=1).tolist())

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

    reports_dir = 'reports'
    os.makedirs(reports_dir, exist_ok=True)

    # save histories and plots
    # convert history_ft/history to history-like object for plot fn
    class SimpleHistory:
        def __init__(self, h):
            self.history = h
    plot_and_save_history(SimpleHistory(hist_combined), reports_dir)

    overall = save_reports(y_true, y_pred, CLASS_NAMES, reports_dir)
    print("Evaluation overall:", overall)

    # Save models: .keras and .h5
    model.save('brain_tumor_inceptionv3.keras')
    try:
        model.save('brain_tumor_inceptionv3.h5')
    except Exception:
        # some TF builds don't allow direct h5 save; fallback to using keras API
        model.save('brain_tumor_inceptionv3.h5', save_format='h5')

    print('Saved: best_inceptionv3.keras, brain_tumor_inceptionv3.keras, brain_tumor_inceptionv3.h5')

    # Export a clean SavedModel (no random augmentation) for serving:
    export_inputs = keras.Input(shape=IMG_SIZE + (3,), name="image")
    y = tf.keras.applications.inception_v3.preprocess_input(export_inputs)
    y = base(y, training=False)  # reuse trained base
    y = layers.GlobalAveragePooling2D()(y)
    y = layers.Dense(len(CLASS_NAMES), activation='softmax', name="predictions")(y)
    export_model = keras.Model(export_inputs, y, name="brain_tumor_inceptionv3_export")

    # copy final dense weights from trained model
    orig_dense = None
    for layer in model.layers[::-1]:
        if isinstance(layer, layers.Dense):
            orig_dense = layer
            break
    if orig_dense is not None:
        export_model.get_layer("predictions").set_weights(orig_dense.get_weights())

    tf.saved_model.save(export_model, "savedmodel_brain_tumor")
    print("SavedModel exported to: savedmodel_brain_tumor/")

if __name__ == '__main__':
    main()
