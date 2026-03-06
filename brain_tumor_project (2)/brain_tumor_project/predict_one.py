import argparse, numpy as np, tensorflow as tf
from tensorflow import keras
from PIL import Image

CLASS_NAMES = ['glioma','meningioma','pituitary','notumor']
IMG_SIZE = (299, 299)

def preprocess_arr(arr):
    return tf.keras.applications.inception_v3.preprocess_input(arr)

def predict_image(model_path, image_path):
    model = keras.models.load_model(model_path)
    img = Image.open(image_path).convert('RGB').resize(IMG_SIZE)
    arr = np.array(img)[None, ...].astype(np.float32)
    arr = preprocess_arr(arr)
    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='best_inceptionv3.keras')
    p.add_argument('--image', required=True)
    args = p.parse_args()
    label, probs = predict_image(args.model, args.image)
    print('Predicted:', label)
    print('Probabilities:', probs)

if __name__ == '__main__':
    main()
