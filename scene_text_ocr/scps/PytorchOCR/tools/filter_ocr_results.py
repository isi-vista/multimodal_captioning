import tensorflow as tf
import os
import sys
import re
import pickle
from os.path import basename, splitext

def init_args():
    import argparse

    parser = argparse.ArgumentParser(description="Filtering OCR results")
    parser.add_argument("--input_file", required=True, type=str, help="input ocr file")
    parser.add_argument("--tokenizer", required=True, type=str, help="input tokenizer")
    parser.add_argument("--conf_threshold", default=0.6, type=float, help="score confidence threshold")
    parser.add_argument("--min_words", default=5, type=int, help="minimum words required for an image")
    parser.add_argument(
        "--output_file", required=True, type=str, help="output ocr file"
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = init_args()

    with open(args.tokenizer, "rb") as fh:
        tokenizer = pickle.load(fh)

    # Load OCR results with scores
    trans = {}
    with open(args.input_file, "r") as fh:
        while True:
            # word line
            line = fh.readline()
            res = line.strip().split()
            if len(res) > 1:
                image = line.strip().split()[0]
                word = line.strip().split()[1]
                name = splitext(basename(image))[0]
                name = re.sub("_\d+$", "", name)

                # score line
                line = fh.readline()
                score = float(line.strip().split()[1])

                if score > args.conf_threshold:
                    if name in trans:
                        trans[name].append((word, score))
                    else:
                        trans[name] = [(word, score)]
            else:
                line = fh.readline()

            if not line:
                break

    seqs, ocrs = {}, {}
    for image in trans:
        if len(trans[image]) >= args.min_words:
            res = sorted(trans[image], key=lambda x: x[1], reverse=True)
            ocrs[image] = " ".join(list(zip(*res))[0][0:args.min_words])
    seqs = tokenizer.texts_to_sequences(ocrs.values())
    seqs = tf.keras.preprocessing.sequence.pad_sequences(seqs, padding="post", maxlen=5)
    seqs = dict(zip(ocrs.keys(), seqs))
    
    with open(args.output_file, "wb") as fh:
        pickle.dump([seqs, ocrs], fh, protocol=pickle.HIGHEST_PROTOCOL)

