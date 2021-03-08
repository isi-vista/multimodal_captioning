import os, sys
from os.path import join, splitext, basename
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

def load_annotation(annotation_file, image_dir):
    with open(annotation_file, "r") as f:
        annotations = json.load(f)

    trn, val, tst = [[], []], [[], []], [[], []]
    for image in annotations["images"]:
        split = image["split"]
        image_path = join(image_dir, image["filepath"], image["filename"])
        for sentence in image["sentences"]:
            if split == "test":
                tst[0].append(image_path)
                tst[1].append(" ".join(sentence["tokens"]))
            elif split == "val":
                val[0].append(image_path)
                val[1].append(" ".join(sentence["tokens"]))
            elif split == "train" or "restval":
                trn[0].append(image_path)
                trn[1].append(" ".join(sentence["tokens"]))

    return [
        list(zip(trn[0], trn[1])),
        list(zip(val[0], val[1])),
        list(zip(tst[0], tst[1])),
    ]


def caption_vectorization(all_caption):

    trn_captions = all_caption[0]
    val_captions = all_caption[1]
    tst_captions = all_caption[2]

    # tokenizer for the training set
    images, captions = map(list, zip(*trn_captions))
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=9000, oov_token="<unk>", filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ '
    )
    tokenizer.fit_on_texts(add_stopwords(captions))
    tokenizer.word_index["<pad>"] = 0
    tokenizer.index_word[0] = "<pad>"

    # Create the tokenized vectors for train set
    # Pad each vector to the max_length of the captions
    # If you do not provide a max_length value, pad_sequences calculates it automatically
    seqs = tokenizer.texts_to_sequences(add_stopwords(captions))
    seqs = tf.keras.preprocessing.sequence.pad_sequences(seqs, padding="post")
    trn_seq = list(zip(images, seqs))

    # create the tokenized vetors for val set
    images, captions = map(list, zip(*val_captions))
    seqs = tokenizer.texts_to_sequences(add_stopwords(captions))
    seqs = tf.keras.preprocessing.sequence.pad_sequences(seqs, padding="post")
    val_seq = list(zip(images, seqs))

    # create the tokenized vetors for test set
    images, captions = map(list, zip(*tst_captions))
    seqs = tokenizer.texts_to_sequences(add_stopwords(captions))
    seqs = tf.keras.preprocessing.sequence.pad_sequences(seqs, padding="post")
    tst_seq = list(zip(images, seqs))

    return tokenizer, trn_seq, val_seq, tst_seq


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Processing MS-COCO (Karpathy splits) caption annotations and image features."
    )
    parser.add_argument("--input-annotation-file", dest="annotation_file")
    parser.add_argument("--input-image-dir", dest="image_dir")
    parser.add_argument("--output-dir", dest="output_dir")
    args = parser.parse_args()

    caps = load_annotation(args.annotation_file, args.image_dir)

    files = ["trn_img_key.pkl", "val_img_key.pkl", "test_img_key.pkl"]
    for f, c in zip(files, caps):
        # key (file path) of images
        keys = set(list(zip(*c))[0])
        with open(join(args.output_dir, f), "wb") as fh:
            pickle.dump(keys, fh, protocol=pickle.HIGHEST_PROTOCOL)

    files = ["trn_caption.pkl", "val_caption.pkl", "test_caption.pkl"]
    for f, c in zip(files, caps):
        with open(join(args.output_dir, f), "wb") as fh:
            pickle.dump(c, fh, protocol=pickle.HIGHEST_PROTOCOL)

    tokenizer, trn_seq, val_seq, test_seq = caption_vectorization(caps)

    with open(join(args.output_dir, "tokenizer.pkl"), "wb") as fh:
        pickle.dump(tokenizer, fh, protocol=pickle.HIGHEST_PROTOCOL)

    files = ["trn_seqs.pkl", "val_seqs.pkl", "test_seqs.pkl"]
    for f, s in zip(files, [trn_seq, val_seq, test_seq]):
        with open(join(args.output_dir, f), "wb") as fh:
            pickle.dump(s, fh, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    sys.exit(main())
