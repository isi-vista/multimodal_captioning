#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import os
import tensorflow as tf
import cv2
from tqdm import tqdm
import pickle

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", help="Run prediction on given images. ")
    parser.add_argument("--output-filename", help="name of featex files")
    parser.add_argument("--output-dir", help="Save featex to dir")
    args = parser.parse_args()
    
    # load image paths
    with open(args.predict, "rb") as fh:
        image_paths = pickle.load(fh)

    # load models
    image_model = tf.keras.applications.InceptionV3(
        include_top=False, weights="imagenet"
    )
    fetex_model = tf.keras.Model(image_model.input, image_model.layers[-1].output)

    # parallelize feature extraction jobs
    img_dataset = tf.data.Dataset.from_tensor_slices(sorted(image_paths))
    img_dataset = img_dataset.map(
        load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
    ).batch(16)

    feats = {}
    for img, path in img_dataset:
        batch_features = fetex_model(img)
        batch_features = tf.reshape(
            batch_features, (batch_features.shape[0], -1, batch_features.shape[3])
        )

        for bf, p in zip(batch_features, path):
            image_file_name = p.numpy().decode("utf-8")
            feats[image_file_name] = bf.numpy()

    with open(os.path.join(args.output_dir, args.output_filename), "wb") as fh:
        pickle.dump(feats, fh, protocol=pickle.HIGHEST_PROTOCOL)
