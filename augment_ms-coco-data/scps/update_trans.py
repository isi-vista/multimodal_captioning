import os, sys
from os.path import join, splitext, basename
import argparse
import pickle
from tqdm import tqdm
import multiprocessing as mp
from evaluation import PTBTokenizer, Bleu
import copy


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Add confidence for each ground truth for MS-COCO data."
    )
    parser.add_argument("--input-trans-file", dest="trans_file")
    parser.add_argument("--conf-dict-file", dest="conf_dict")
    parser.add_argument("--output-file", dest="output_file")
    args = parser.parse_args()

    dict_data = {}
    with open(args.conf_dict, "rb") as fh:
        dict_data = pickle.load(fh)

    with open(args.trans_file, "rb") as fh:
        trans = pickle.load(fh)

    # trans update
    dict_data_cp = copy.deepcopy(dict_data)
    for i, item in enumerate(trans[0]):
        img = item[0]
        score = dict_data_cp[img].pop(0)
        trans[0][i] = item + (score,)

    # bpe update
    dict_data_cp = copy.deepcopy(dict_data)
    for i, item in enumerate(trans[1]):
        img = item[0]
        score = dict_data_cp[img].pop(0)
        trans[1][i] = item + (score,)

    # tokens update
    dict_data_cp = copy.deepcopy(dict_data)
    for i, item in enumerate(trans[2]):
        img = item[0]
        score = dict_data_cp[img].pop(0)
        trans[2][i] = item + (score,)

    with open(args.output_file, "wb") as fh:
        pickle.dump(
            [trans[0], trans[1], trans[2]], fh, protocol=pickle.HIGHEST_PROTOCOL
        )


if __name__ == "__main__":
    sys.exit(main())
