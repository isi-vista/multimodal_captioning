import os, sys
from os.path import join, splitext, basename
import argparse
import pickle
from tqdm import tqdm
import multiprocessing as mp
from evaluation import PTBTokenizer, Bleu
import copy


def compute_score(gen, gts):
    # score_gts = PTBTokenizer.tokenize(gts)
    # score_gen = PTBTokenizer.tokenize(gen)
    bleu_scores, _ = Bleu().compute_score(gts, gen)
    return bleu_scores[3]


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Add confidence for each ground truth for MS-COCO data."
    )
    parser.add_argument("--input-trans-file", dest="trans_file")
    parser.add_argument("--start-index", type=int, dest="start")
    parser.add_argument("--end-index", type=int, dest="end")
    parser.add_argument("--output-file", dest="output_file")
    args = parser.parse_args()

    trans_data = {}
    with open(args.trans_file, "rb") as fh:
        data = pickle.load(fh)

        for item in data[0]:  # (image_name, eng_trans, de_trans, score)
            if item[0] in trans_data:
                trans_data[item[0]].append(item[1])
            else:
                trans_data[item[0]] = [item[1]]

    trans_scores = {}
    trans_keys = list(trans_data.keys())
    trans_keys.sort()
    for img in tqdm(trans_keys):
        params = []
        sentences = [
            copy.deepcopy(trans_data[img]) for _ in range(len(trans_data[img]))
        ]
        for i, s in enumerate(sentences):
            gts, gen = {}, {}
            g = s.pop(i)
            gen[i] = [g]
            gts[i] = s
            params.append((gen, gts))

        with mp.Pool(mp.cpu_count()) as p:
            scores = p.starmap(compute_score, params)

        trans_scores[img] = scores

    with open(args.output_file, "wb") as fh:
        pickle.dump(trans_scores, fh, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    sys.exit(main())
