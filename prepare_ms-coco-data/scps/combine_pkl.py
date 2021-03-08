import os, sys
from os.path import join, splitext, basename
import argparse
import json
import pickle
import tensorflow as tf

# Example:
# python -u combine_pkl.py --input-pickle-files ./expts/feats/faster-rcnn/1.pkl ./expts/feats/faster-rcnn/2.pkl --output-pickle-file ./expts/feats/faster-rcnn/trn_cascade_fastrcnn_featex.pkl

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Combine multiple pickle files into a single one."
    )
    parser.add_argument("--input-pickle-files", dest="pickle_files", nargs="+" )
    parser.add_argument("--output-pickle-file", dest="out_file")
    args = parser.parse_args()

    out_content = {}
    for file in args.pickle_files:
        print(file, flush=True)
        with open(file, "rb") as fh:
            content = pickle.load(fh)
            out_content.update(content)

    with open(args.out_file, "wb") as fh:
        pickle.dump(out_content, fh, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    sys.exit(main())
