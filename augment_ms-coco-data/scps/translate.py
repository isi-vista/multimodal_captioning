import os, sys
from os.path import join, splitext, basename
import argparse
import numpy as np
import json
import pickle
import tensorflow as tf
from tqdm import tqdm
from learn_bpe import learn_bpe
from apply_bpe import apply_bpe
from pathos.multiprocessing import ProcessingPool as Pool
from fairseq.models.transformer import TransformerModel


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


def trans_bpe(input_raw_trans):
    trn_img, trn_en_trans, trn_de_trans, trn_scores = map(
        list, zip(*input_raw_trans[0])
    )
    val_img, val_en_trans, val_de_trans, val_scores = map(
        list, zip(*input_raw_trans[1])
    )
    tst_img, tst_en_trans, tst_de_trans, tst_scores = map(
        list, zip(*input_raw_trans[2])
    )

    codes = learn_bpe(trn_en_trans + trn_de_trans, 10000, num_workers=10)

    trn_en_trans = apply_bpe(codes, trn_en_trans, num_workers=10)
    trn_de_trans = apply_bpe(codes, trn_de_trans, num_workers=10)
    val_en_trans = apply_bpe(codes, val_en_trans, num_workers=10)
    val_de_trans = apply_bpe(codes, val_de_trans, num_workers=10)
    tst_en_trans = apply_bpe(codes, tst_en_trans, num_workers=10)
    tst_de_trans = apply_bpe(codes, tst_de_trans, num_workers=10)

    return (
        codes,
        [
            list(zip(trn_img, trn_en_trans, trn_de_trans, trn_scores)),
            list(zip(val_img, val_en_trans, val_de_trans, val_scores)),
            list(zip(tst_img, tst_en_trans, tst_de_trans, tst_scores)),
        ],
    )


def trans_tokenization(all_trans):
    trn_captions = all_trans[0]
    val_captions = all_trans[1]
    tst_captions = all_trans[2]

    # tokenizer for the training set
    images, en_captions, de_captions, scores = map(list, zip(*trn_captions))
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=9000, oov_token="<unk>", filters='!"#$%&()*+.,-/:;=?[\]^_`{|}~ '
    )
    tokenizer.fit_on_texts(add_stopwords(en_captions + de_captions))
    tokenizer.word_index["<pad>"] = 0
    tokenizer.index_word[0] = "<pad>"

    # Create the tokenized vectors for train set
    # Pad each vector to the max_length of the captions
    # If you do not provide a max_length value, pad_sequences calculates it automatically
    en_seqs = tokenizer.texts_to_sequences(add_stopwords(en_captions))
    en_seqs = tf.keras.preprocessing.sequence.pad_sequences(
        en_seqs, padding="post", maxlen=128
    )
    de_seqs = tokenizer.texts_to_sequences(add_stopwords(de_captions))
    de_seqs = tf.keras.preprocessing.sequence.pad_sequences(de_seqs, padding="post")
    trn_seq = list(zip(images, en_seqs, de_seqs, scores))

    # create the tokenized vetors for val set
    images, en_captions, de_captions, scores = map(list, zip(*val_captions))
    en_seqs = tokenizer.texts_to_sequences(add_stopwords(en_captions))
    en_seqs = tf.keras.preprocessing.sequence.pad_sequences(
        en_seqs, padding="post", maxlen=128
    )
    de_seqs = tokenizer.texts_to_sequences(add_stopwords(de_captions))
    de_seqs = tf.keras.preprocessing.sequence.pad_sequences(de_seqs, padding="post")
    val_seq = list(zip(images, en_seqs, de_seqs, scores))

    # create the tokenized vetors for test set
    images, en_captions, de_captions, scores = map(list, zip(*tst_captions))
    en_seqs = tokenizer.texts_to_sequences(add_stopwords(en_captions))
    en_seqs = tf.keras.preprocessing.sequence.pad_sequences(
        en_seqs, padding="post", maxlen=128
    )
    de_seqs = tokenizer.texts_to_sequences(add_stopwords(de_captions))
    de_seqs = tf.keras.preprocessing.sequence.pad_sequences(de_seqs, padding="post")
    tst_seq = list(zip(images, en_seqs, de_seqs, scores))

    return tokenizer, [trn_seq, val_seq, tst_seq]


def en2de(model_dir, input_en_trans):
    model = TransformerModel.from_pretrained(
        model_dir + "/wmt19.en-de.joined-dict.ensemble",
        checkpoint_file="model4.pt",
        bpe="fastbpe",
        bpe_codes=model_dir + "/wmt19.en-de.joined-dict.ensemble/bpecodes",
        num_workers=10,
    )

    multi_trans = []

    for trans in input_en_trans:
        results = []
        images, en_trans = map(list, zip(*trans))
        chunks = np.array_split(en_trans, 100)

        for c in tqdm(chunks):
            de_trans = model.translate(c)
            results += de_trans
        de_trans, scores = zip(*results)

        multi_trans.append(list(zip(images, en_trans, de_trans, scores)))

    return multi_trans


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Processing MS-COCO (Karpathy splits) caption annotations and image features."
    )
    parser.add_argument("--input-annotation-file", dest="annotation_file")
    parser.add_argument("--input-image-dir", dest="image_dir")
    parser.add_argument("--fairseq-model-dir", dest="fairseq_model_dir")
    parser.add_argument("--output-dir", dest="output_dir")
    args = parser.parse_args()

    en_raw_trans = load_annotation(args.annotation_file, args.image_dir)

    multi_raw_trans = en2de(args.fairseq_model_dir, en_raw_trans)

    bpe_codes, bpe_trans = trans_bpe(multi_raw_trans)

    tokenizer, seqs = trans_tokenization(bpe_trans)

    # save files
    with open(join(args.output_dir, "bpe.codes.pkl"), "wb") as fh:
        pickle.dump(bpe_codes, fh, protocol=pickle.HIGHEST_PROTOCOL)

    with open(join(args.output_dir, "tokenizer.pkl"), "wb") as fh:
        pickle.dump(tokenizer, fh, protocol=pickle.HIGHEST_PROTOCOL)

    prefix = ["train", "val", "test"]
    for p, f1, f2, f3 in zip(prefix, multi_raw_trans, bpe_trans, seqs):
        with open(join(args.output_dir, p + ".trans.pkl"), "wb") as fh:
            pickle.dump([f1, f2, f3], fh, protocol=pickle.HIGHEST_PROTOCOL)

    files = ["train.image.list.pkl", "val.image.list.pkl", "test.image.list.pkl"]
    for f, c in zip(files, multi_raw_trans):
        # key (file path) of images
        keys = set(list(zip(*c))[0])
        with open(join(args.output_dir, f), "wb") as fh:
            pickle.dump(keys, fh, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    sys.exit(main())
