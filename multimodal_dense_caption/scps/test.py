import os, sys
import argparse
import tensorflow as tf
import pickle
from data import dataset
from models.transformer import Transformer
from evaluation import PTBTokenizer, Cider, utils
from trainer import evaluate_metrics


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="Image Captioning Test")
    parser.add_argument("--exp_name", type=str, default="base_transformer")
    parser.add_argument(
        "--feature_files",
        required=True,
        help="feature files for train, val, and test set",
    )
    parser.add_argument(
        "--raw_caption_files",
        required=True,
        help="raw caption files for train, val, and test set",
    )
    parser.add_argument(
        "--tokenized_caption_files",
        required=True,
        help="tokenized caption files for train, val, and test set",
    )
    parser.add_argument("--tokenizer_file", required=True)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--saved_checkpoint", required=True)
    parser.add_argument("--output_file", type=str, default="output.pkl")
    args = parser.parse_args()

    dataloader_val = dataset.Multi30kDataLoader(
        args.caption_files, args.tokenized_caption_files, args.feature_files[1],
    )

    # tokenizer
    with open(args.tokenizer_file, "rb") as fh:
        tokenizer = pickle.load(fh)

    # setup transformer model
    source_vocab_size = 4652
    target_vocab_size = 4652
    max_len = 50000
    d_model = 512
    model = Transformer(
        source_vocab_size=source_vocab_size,
        target_vocab_size=target_vocab_size,
        max_len=max_len,
    )

    # setup checkpoint
    ckpt = tf.train.Checkpoint(model=model,)

    ckpt_file = "%s/%s/best" % (args.saved_checkpoint, args.exp_name)

    if os.path.exists(ckpt_file + ".index"):
        ckpt.restore(ckpt_file).expect_partial()
        print("Restored checkpoint done.")

    scores, gts, gen = evaluate_metrics(model, dataloader.get(), tokenizer, 5, 0)

    print("Evaluation scores", scores)

    with open(args.output_file, "wb") as fh:
        pickle.dump([scores, gts, gen], fh, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    sys.exit(main())
