# eval_only.py
import os, argparse, json, numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt, seaborn as sns, csv

def try_load_model(path):
    if path and os.path.exists(path):
        print("Loading model:", path)
        return keras.models.load_model(path)
    # try default name
    if os.path.exists("brain_tumor_inceptionv3.keras"):
        print("Loading model: brain_tumor_inceptionv3.keras")
        return keras.models.load_model("brain_tumor_inceptionv3.keras")
    raise FileNotFoundError(f"Model path does not exist: {path or 'brain_tumor_inceptionv3.keras'}")

def build_dataset(test_dir, img_size, batch_size=32):
    ds = tf.keras.preprocessing.image_dataset_from_directory(test_dir, labels='inferred', label_mode='int',
                                                             image_size=img_size, batch_size=batch_size, shuffle=False)
    return ds

def save_reports(y_true, y_pred, class_names, out_folder):
    os.makedirs(out_folder, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted label'); plt.ylabel('True label'); plt.title('Confusion Matrix')
    plt.savefig(os.path.join(out_folder, 'confusion_matrix.png'))
    plt.close()
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    with open(os.path.join(out_folder, 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(classification_report(y_true, y_pred, target_names=class_names))
    with open(os.path.join(out_folder, 'classification_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    csv_path = os.path.join(out_folder, 'per_class_metrics.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['class','precision','recall','f1-score','support'])
        for c in class_names:
            r = report.get(c, {})
            writer.writerow([c, r.get('precision',0), r.get('recall',0), r.get('f1-score',0), r.get('support',0)])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, help='path to .h5 or .keras model (optional)')
    parser.add_argument('--test_dir', default='data/test', help='test dataset directory')
    parser.add_argument('--img_size', type=int, nargs=2, default=[299,299])
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    model = try_load_model(args.model)
    img_size = tuple(args.img_size)
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(args.test_dir, labels='inferred', label_mode='int', image_size=img_size, batch_size=args.batch_size, shuffle=False)
    class_names = test_ds.class_names
    print("Found classes:", class_names)

    y_true = []
    y_pred = []
    for x,y in test_ds:
        preds = model.predict(x)
        y_true.extend(y.numpy().tolist())
        y_pred.extend(np.argmax(preds, axis=1).tolist())

    out = 'reports'
    os.makedirs(out, exist_ok=True)
    save_reports(y_true, y_pred, class_names, out)

    acc = np.mean(np.array(y_true) == np.array(y_pred))
    print("=== Overall metrics ===")
    print("Accuracy:", round(float(acc),4))
    # print per-class summary path
    print("Reports saved to folder:", os.path.abspath(out))
