import os, sys
from os.path import join, exists
import argparse
import json
import pickle
import tensorflow as tf
from learn_bpe import learn_bpe
from apply_bpe import apply_bpe
from pathos.multiprocessing import ProcessingPool as Pool


def add_stopwords(sentences: list):
    def adding(text):
        return "<bos> " + text + " <eos>"

    with Pool() as p:
        sentences = p.map(adding, sentences)

    return sentences


def tokenization(tokenizer, en_trans, de_trans, num_words=30000):
    en_images, en_captions = map(list, zip(*en_trans))
    de_images, de_captions = map(list, zip(*de_trans))

    if tokenizer is None:
        tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=30000, oov_token="<unk>", filters='!"#$%&()*+.,-/:;=?[\]^_`{|}~ '
        )
        tokenizer.fit_on_texts(add_stopwords(en_captions + de_captions))
        tokenizer.word_index["<pad>"] = 0
        tokenizer.index_word[0] = "<pad>"

    # Create the tokenized vectors for train set
    # Pad each vector to the max_length of the captions
    # If you do not provide a max_length value, pad_sequences calculates it automatically
    en_seqs = tokenizer.texts_to_sequences(add_stopwords(en_captions))
    en_seqs = tf.keras.preprocessing.sequence.pad_sequences(en_seqs, padding="post")
    en_seqs = list(zip(en_images, en_seqs))

    de_seqs = tokenizer.texts_to_sequences(add_stopwords(de_captions))
    de_seqs = tf.keras.preprocessing.sequence.pad_sequences(de_seqs, padding="post")
    de_seqs = list(zip(de_images, de_seqs))

    return tokenizer, en_seqs, de_seqs


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
    parser.add_argument("--codes")
    parser.add_argument("--en_trans", nargs="+")
    parser.add_argument("--de_trans", nargs="+")
    parser.add_argument("--image_dir")
    parser.add_argument("--output_prefix")
    args = parser.parse_args()

    # images
    image_list = load_image_list(args.image_dir, args.image_list)

    # raw trans
    en_raw_trans = load_trans(image_list, args.en_trans)
    de_raw_trans = load_trans(image_list, args.de_trans)

    # vocab for BPE
    if exists(args.codes):
        with open(args.codes, "rb") as fh:
            codes = pickle.load(fh)
    else:
        codes = None

    # tokenized trans
    if exists(args.tokenizer):
        with open(args.tokenizer, "rb") as fh:
            tokenizer = pickle.load(fh)
    else:
        tokenizer = None

    # BPE learning
    en_images, en_trans = map(list, zip(*en_raw_trans))
    de_images, de_trans = map(list, zip(*de_raw_trans))
    merged_trans = en_trans + de_trans

    if codes is None:
        codes = learn_bpe(merged_trans, 10000, num_workers=10)

    en_trans = apply_bpe(codes, en_trans, num_workers=10)
    de_trans = apply_bpe(codes, de_trans, num_workers=10)

    tokenizer, en_seqs, de_seqs = tokenization(
        tokenizer, list(zip(en_images, en_trans)), list(zip(de_images, de_trans)), 30000
    )

    # save files
    with open(args.output_prefix + ".codes.pkl", "wb") as fh:
        pickle.dump(codes, fh, protocol=pickle.HIGHEST_PROTOCOL)

    with open(args.output_prefix + ".tokenizer.pkl", "wb") as fh:
        pickle.dump(tokenizer, fh, protocol=pickle.HIGHEST_PROTOCOL)

    with open(args.output_prefix + ".trans.pkl", "wb") as fh:
        pickle.dump(
            [en_raw_trans, en_trans, en_seqs, de_raw_trans, de_trans, de_seqs],
            fh,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    with open(args.output_prefix + ".image.list.pkl", "wb") as fh:
        pickle.dump(image_list, fh, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    sys.exit(main())
