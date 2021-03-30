import sys
import pickle
from os.path import basename, splitext

def init_args():
    import argparse

    parser = argparse.ArgumentParser(description="Filtering OCR results")
    parser.add_argument("--input_key_file")
    parser.add_argument("--input_visual_feat_file")
    parser.add_argument("--input_image_class_file")
    parser.add_argument("--input_text_file")
    parser.add_argument("--output_visual_feat_file")
    parser.add_argument("--output_image_class_file")
    parser.add_argument("--output_text_file")
    args = parser.parse_args()
    return args


def main(argv=None):
    args = init_args()

    with open(args.input_key_file, "rb") as fh:
        keys = pickle.load(fh)[0].keys()

    # text 
    with open(args.input_text_file, "rb") as fh1, open(args.output_text_file, "wb") as fh2:
        t1, t2, t3 = pickle.load(fh1)

        tr1,tr2,tr3 = [], [], []
        for i, item in enumerate(zip(t1, t2, t3)):
            name = splitext(basename(item[0][0]))[0]
            if name in keys:
                tr1.append( (name,) + item[0][1:] )
                tr2.append( (name,) + item[1][1:] )
                tr3.append( (name,) + item[2][1:] )

        pickle.dump([tr1, tr2, tr3], fh2, protocol=pickle.HIGHEST_PROTOCOL)

    # objects visual feature
    with open(args.input_visual_feat_file, "rb") as fh1, open(args.output_visual_feat_file, "wb") as fh2:
        f = {}
        feat = pickle.load(fh1) 
        for key in feat:
            name = splitext(basename(key))[0]
            if name in keys:
                f[name] = feat[key]

        pickle.dump(f, fh2, protocol=pickle.HIGHEST_PROTOCOL)

    # image class feature
    with open(args.input_image_class_file, "rb") as fh1, open(args.output_image_class_file, "wb") as fh2:
        l, c = {}, {}
        labels, classes = pickle.load(fh1)
        for key in labels:
            name = splitext(basename(key))[0]
            if name in keys:
                l[name] = labels[key]
                c[name] = classes[key]

        pickle.dump([l, c], fh2, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    sys.exit(main())
