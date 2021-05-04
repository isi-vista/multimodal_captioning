import tensorflow as tf
import numpy as np
import random
import pickle
from os.path import basename, splitext
from sklearn.utils import shuffle


class COCODataLoader:
    def __init__(self, caption_file, feature_file, class_file, dense_cap_file, batch_size=10):
        self.load_dataset(caption_file, feature_file, class_file, dense_cap_file)
        self.data = self.refresh(batch_size)

    def get(self):
        return self.data

    def load_dataset(self, caption_file, feature_file, class_file, dense_cap_file):
        # load images (keys) and tokenized captions
        with open(caption_file, "rb") as fh:
            captions = pickle.load(fh)

        self.images, self.en_seqs, self.de_seqs, self.scores = map(
            list, zip(*captions[2])
        )

        # create images -> captions dictionary
        self.de_caps = {key: [] for key in self.images}
        for im, en, de, score in captions[0]:
            self.de_caps[im].append(de)

        # load features
        with open(feature_file, "rb") as fh:
            features = pickle.load(fh)
        self.features = dict(features)

        # load classes of objects
        with open(class_file, "rb") as fh:
            classes = pickle.load(fh)
        self.classes = classes[0]

        # load dense captions
        with open(dense_cap_file, "rb") as fh:
            dense_caps = pickle.load(fh)
        self.dense_caps = {}
        # hard-coded dense captions number as 3
        for im in dense_caps:
            self.dense_caps[im] = random.choices(list(dense_caps[im][2]), k=3)

    # create database with map func
    def dataset_map_func(self, im, en_seq, de_seq, score):
        feat = [obj["feature"] for obj in self.features[im.decode("utf-8")]]
        cls = self.classes[im.decode("utf-8")]
        caps = self.de_caps[im.decode("utf-8")][0:5]
        dense_caps = self.dense_caps[im.decode("utf-8")]
        return np.array(feat), cls, dense_caps, en_seq, de_seq, caps, score

    def refresh(self, batch_size=64, sample_num=None):
        images, en_seqs, de_seqs, scores = shuffle(
            self.images, self.en_seqs, self.de_seqs, self.scores, n_samples=sample_num
        )

        # dataset
        data = tf.data.Dataset.from_tensor_slices((images, en_seqs, de_seqs, scores))
        data = data.map(
            lambda im, en_seq, de_seq, score: tf.numpy_function(
                self.dataset_map_func,
                [im, en_seq, de_seq, score],
                [tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.string, tf.float32],
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

        # shuffle the dataset
        data = data.shuffle(buffer_size=5000, reshuffle_each_iteration=True).batch(
            batch_size, drop_remainder=True
        )
        data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        # the output dataloader
        dataloader = dict()
        dataloader["data"] = data
        dataloader["captions"] = self.de_caps
        dataloader["batch_size"] = batch_size
        dataloader["data_steps"] = len(images) // batch_size

        return dataloader


class Multi30kDataLoader:
    def __init__(self, caption_file, feature_file, class_file, dense_cap_file, batch_size=10):
        self.load_dataset(caption_file, feature_file, class_file, dense_cap_file)
        self.data = self.refresh(batch_size)

    def get(self):
        return self.data

    def load_dataset(self, caption_file, feature_file, class_file, dense_cap_file):
        # load images (keys) and captions
        with open(caption_file, "rb") as fh:
            captions = pickle.load(fh)
        self.images, self.en_caps = map(list, zip(*captions[0]))
        _, self.en_seqs = map(list, zip(*captions[2]))
        _, self.de_caps = map(list, zip(*captions[3]))
        _, self.de_seqs = map(list, zip(*captions[5]))

        # create images -> captions dictionary
        self.dict_images_en_caps = {key: [] for key in self.images}
        for image, cap in zip(self.images, self.en_caps):
            self.dict_images_en_caps[image].append(cap)

        self.dict_images_en_seqs = {key: [] for key in self.images}
        for image, cap in zip(self.images, self.en_seqs):
            self.dict_images_en_seqs[image].append(cap)

        self.dict_images_de_caps = {key: [] for key in self.images}
        for image, cap in zip(self.images, self.de_caps):
            self.dict_images_de_caps[image].append(cap)

        self.dict_images_de_seqs = {key: [] for key in self.images}
        for image, cap in zip(self.images, self.de_seqs):
            self.dict_images_de_seqs[image].append(cap)

        # load features
        with open(feature_file, "rb") as fh:
            features = pickle.load(fh)
        self.features = dict(features)

        # load classes of objects
        with open(class_file, "rb") as fh:
            classes = pickle.load(fh)
        self.classes = classes[0]

        # load dense captions
        with open(dense_cap_file, "rb") as fh:
            dense_caps = pickle.load(fh)
        self.dense_caps = {}
        # hard-coded dense captions number as 3
        for im in dense_caps:
            self.dense_caps[im] = random.choices(list(dense_caps[im][2]), k=3)

    # create database with map func
    def dataset_map_func(self, image):
        feat = [obj["feature"] for obj in self.features[image.decode("utf-8")]]
        cls = self.classes[image.decode("utf-8")]
        en_seqs = self.dict_images_en_seqs[image.decode("utf-8")]
        de_seqs = self.dict_images_de_seqs[image.decode("utf-8")]
        de_caps = self.dict_images_de_caps[image.decode("utf-8")]
        dense_caps = self.dense_caps[image.decode("utf-8")]
        return np.array(feat), cls, dense_caps, en_seqs[0], de_seqs[0], de_caps[0], np.float32(1.0)

    def refresh(self, batch_size=10, sample_num=None):
        images = shuffle(self.images, n_samples=sample_num)

        # target tokenized sequences dataset
        data = tf.data.Dataset.from_tensor_slices(images)
        data = data.map(
            lambda im: tf.numpy_function(
                self.dataset_map_func,
                [im],
                [tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.string, tf.float32],
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

        # shuffle the dataset
        data = data.shuffle(buffer_size=5000, reshuffle_each_iteration=True).batch(
            batch_size, drop_remainder=True
        )
        data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        # the output dataloader
        dataloader = dict()
        dataloader["data"] = data
        dataloader["captions"] = self.de_caps
        dataloader["batch_size"] = batch_size
        dataloader["data_steps"] = len(images) // batch_size

        return dataloader
