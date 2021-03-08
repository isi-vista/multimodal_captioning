import os, sys
from os.path import join, exists
import argparse
import json
import pickle
import tensorflow as tf
from pathos.multiprocessing import ProcessingPool as Pool


def add_stopwords(sentences: list):
    def adding(text):
        return "<bos> " + text + " <eos>"

    with Pool() as p:
        sentences = p.map(adding, sentences)

    return sentences


def tokenization(tokenizer, trans, num_words=10000):
    images, captions = map(list, zip(*trans))

    if tokenizer is None:
        tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=num_words, oov_token="<unk>", filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ '
        )
        tokenizer.fit_on_texts(add_stopwords(captions))
        tokenizer.word_index["<pad>"] = 0
        tokenizer.index_word[0] = "<pad>"

    # Create the tokenized vectors for train set
    # Pad each vector to the max_length of the captions
    # If you do not provide a max_length value, pad_sequences calculates it automatically
    seqs = tokenizer.texts_to_sequences(add_stopwords(captions))
    seqs = tf.keras.preprocessing.sequence.pad_sequences(seqs, padding="post")
    seqs = list(zip(images, seqs))

    return tokenizer, seqs


def load_image_list(image_dir, list_file):
    image_list = []
    with open(list_file, "r") as fh:
        for line in fh:
            image_list.append(join(image_dir, line.strip()))
    return image_list


def load_trans(image_list, trans_files):
    trans_all = []

    for file in trans_files:
        trans = []
        with open(file, "r") as fh:
            for line in fh:
                text = line.rstrip()
                trans.append(text)

        trans_all = trans_all + list(zip(image_list, trans))

    return trans_all


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="Processing Multi30K dataset.")
    parser.add_argument("--image_list")
    parser.add_argument("--tokenizer")
    parser.add_argument("--en_trans", nargs="+")
    parser.add_argument("--de_trans", nargs="+")
    parser.add_argument("--image_dir")
    parser.add_argument("--output_prefix")
    args = parser.parse_args()

    # images
    image_list = load_image_list(args.image_dir, args.image_list)

    # raw trans
    en_trans = load_trans(image_list, args.en_trans)
    de_trans = load_trans(image_list, args.de_trans)

    # tokenized trans
    if exists(args.tokenizer):
        with open(args.tokenizer, "rb") as fh:
            tokenizer = pickle.load(fh)
            en_tokenizer = tokenizer[0]
            de_tokenizer = tokenizer[1]
    else:
        en_tokenizer = None
        de_tokenizer = None

    en_tokenizer, en_seqs = tokenization(en_tokenizer, en_trans, 5000)
    de_tokenizer, de_seqs = tokenization(de_tokenizer, de_trans, 10000)

    # save files
    with open(args.output_prefix + ".tokenizer.pkl", "wb") as fh:
        pickle.dump([en_tokenizer, de_tokenizer], fh, protocol=pickle.HIGHEST_PROTOCOL)

    with open(args.output_prefix + ".trans.pkl", "wb") as fh:
        pickle.dump(
            [en_trans, en_seqs, de_trans, de_seqs], fh, protocol=pickle.HIGHEST_PROTOCOL
        )

    with open(args.output_prefix + ".image.list.pkl", "wb") as fh:
        pickle.dump(image_list, fh, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    sys.exit(main())
