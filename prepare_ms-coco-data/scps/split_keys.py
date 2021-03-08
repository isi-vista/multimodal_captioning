import os, sys
from os.path import join, splitext, basename
import argparse
import json
import pickle
import tensorflow as tf

# Example:
# python -u split_key.py --input-key-file ./expts/data/ms-coco/trn_img_key.pkl --chunk-size 11000 --output-dir ./expts/data/ms-coco/trn_img_keys/

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Split large image keys (paths) file into smaller ones."
    )
    parser.add_argument("--input-key-file", dest="key_file")
    parser.add_argument("--chunk-size", dest="chunk_size", type=int)
    parser.add_argument("--output-dir", dest="out_dir")
    args = parser.parse_args()


    with open(args.key_file, "rb") as fh:
        keys = pickle.load(fh)

    def divide_chunks(l, n): 
        # looping till length l 
        for i in range(0, len(l), n):  
            yield l[i:i + n]

    key_list = list(divide_chunks(list(keys), args.chunk_size))

    for idx, l in enumerate(key_list, start=1):
        with open(join(args.out_dir, str(idx))+".pkl", "wb") as fh:
            pickle.dump(set(l), fh, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    sys.exit(main())
