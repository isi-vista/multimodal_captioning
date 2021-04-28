import os, sys
from os.path import join, exists
import argparse
import pickle
import tensorflow as tf
from tqdm import tqdm
from learn_bpe import learn_bpe
from apply_bpe import apply_bpe
from pathos.multiprocessing import ProcessingPool as Pool


def add_stopwords(sentences: list):
    def adding(text):
        return "<bos> " + text + " <eos>"

    with Pool() as p:
        sentences = p.map(adding, sentences)

    return sentences


def tokenization(tokenizer, trans, num_words=30000):
    if tokenizer is None:
        tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=30000, oov_token="<unk>", filters='!"#$%&()*+.,-/:;=?[\]^_`{|}~ '
        )
        tokenizer.fit_on_texts(add_stopwords(trans))
        tokenizer.word_index["<pad>"] = 0
        tokenizer.index_word[0] = "<pad>"

    # Create the tokenized vectors for train set
    # Pad each vector to the max_length of the captions
    # If you do not provide a max_length value, pad_sequences calculates it automatically
    seqs = tokenizer.texts_to_sequences(add_stopwords(trans))
    seqs = tf.keras.preprocessing.sequence.pad_sequences(
        seqs, padding="post", maxlen=128
    )

    return seqs


def load_trans(trans_file):
    with open(trans_file, "rb") as fh:
        data = pickle.load(fh)

    for k in data:
        data[k] = list(data[k].keys())

    return data


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="Processing dataset.")
    parser.add_argument("--tokenizer")
    parser.add_argument("--codes")
    parser.add_argument("--input_trans")
    parser.add_argument("--output_prefix")
    args = parser.parse_args()

    # raw trans
    raw_trans = load_trans(args.input_trans)

    # vocab for BPE
    if exists(args.codes):
        with open(args.codes, "rb") as fh:
            codes = pickle.load(fh)

    # tokenized trans
    if exists(args.tokenizer):
        with open(args.tokenizer, "rb") as fh:
            tokenizer = pickle.load(fh)

    merged_trans = [item for sublist in list(raw_trans.values()) for item in sublist]
    bpe_trans = apply_bpe(codes, merged_trans, num_workers=10)
    seqs = tokenization(tokenizer, bpe_trans, 30000)

    out_trans = {}
    idx = 0
    for key in raw_trans:
        num = len(raw_trans[key])
        out_trans[key] = [raw_trans[key], bpe_trans[idx:idx+num], seqs[idx:idx+num]]
        idx = idx+num

    with open(args.output_prefix + ".bpe.trans.pkl", "wb") as fh:
        pickle.dump(
            out_trans, fh, protocol=pickle.HIGHEST_PROTOCOL,
        )


if __name__ == "__main__":
    sys.exit(main())
