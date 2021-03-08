import os, sys
import argparse
import tensorflow as tf
import pickle
from data import dataset
from models.transformer import Transformer
from evaluation import PTBTokenizer, Cider, utils
from trainer import train_xe, evaluate_loss, evaluate_metrics


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="Image Captioning Training")
    parser.add_argument("--exp_name", type=str, default="base_transformer")
    parser.add_argument(
        "--feature_files",
        required=True,
        nargs="+",
        help="feature files for train, val, and test set",
    )
    parser.add_argument(
        "--class_files",
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "--caption_files",
        required=True,
        nargs="+",
        help="caption files for train, val, and test set",
    )
    parser.add_argument("--tokenizer_file", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--sample_num", type=int, default=29000)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--resume_last", action="store_true")
    parser.add_argument("--saved_checkpoint", required=True)
    parser.add_argument("--logs_folder", type=str, default="logs")
    args = parser.parse_args()

    # check GPU is being used
    if tf.test.gpu_device_name():
        print("Default GPU Device: {}".format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")

    dataloader_coco_train = dataset.COCODataLoader(
        args.caption_files[0], args.feature_files[0], args.class_files[0], args.batch_size
    )
    dataloader_coco_train = dataset.COCODataLoader(
        args.caption_files[1], args.feature_files[1], args.batch_size
    )

    print("load ms-coco done")

    # load train, val, test datasets
    #dataloader_multi30k_train = dataset.Multi30kDataLoader(
    #    args.caption_files[3], args.feature_files[3], args.class_files[3], args.batch_size
    #)
    #dataloader_multi30k_val = dataset.Multi30kDataLoader(
    #    args.caption_files[4], args.feature_files[4], args.class_files[4], args.batch_size
    #)

    print("load multi30k done")

    """
    # prepare cider for training
    cider_train = Cider(
        PTBTokenizer.tokenize(list(dataloader_coco_train.get()["captions"].values()))
    )
    """

    # tokenizer
    with open(args.tokenizer_file, "rb") as fh:
        tokenizer = pickle.load(fh)

    print("Load data completed.")

    # setup transformer model
    source_vocab_size = 4478
    target_vocab_size = 4478
    max_len = 50000
    d_model = 512
    model = Transformer(
        source_vocab_size=source_vocab_size,
        target_vocab_size=target_vocab_size,
        max_len=max_len,
    )

    # Initial conditions
    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )
    start_epoch = tf.Variable(0)
    best_cider = tf.Variable(0.0)

    # setup checkpoint
    ckpt = tf.train.Checkpoint(
        model=model, optimizer=optimizer, epoch=start_epoch, best_cider=best_cider,
    )

    ckpt_file = (
        "%s/%s/last" % (args.saved_checkpoint, args.exp_name)
        if args.resume_last
        else "%s/%s/best" % (args.saved_checkpoint, args.exp_name)
    )

    if os.path.exists(ckpt_file + ".index"):
        ckpt.restore(ckpt_file)
        print(
            "Restored checkpoint, restart training from epoch: %s" % start_epoch.numpy()
        )

    # setup logger
    logger = tf.summary.create_file_writer(
        os.path.join(args.logs_folder, args.exp_name)
    )

    print("Training starts")
    for e in range(start_epoch.numpy(), args.epochs):
        train_loss = train_xe(model, dataloader_coco_train.get(), optimizer, e)
        #train_loss = train_xe(model, dataloader_multi30k_train.get(), optimizer, e)
        with logger.as_default():
            tf.summary.scalar("data/train_loss", train_loss, e)

        """
        # Validation loss
        val_loss = evaluate_loss(model, dataloader_val.get(), e)
        with logger.as_default():
            tf.summary.scalar("data/val_loss", val_loss, e)
        """

        # Validation scores
        scores, _, _ = evaluate_metrics(
            model, dataloader_multi30k_val.get(), tokenizer, 5, e
        )
        print("Validation scores", scores)
        val_cider = scores["CIDEr"]
        with logger.as_default():
            tf.summary.scalar("data/val_cider", val_cider, e)
            tf.summary.scalar("data/val_bleu1", scores["BLEU"][0], e)
            tf.summary.scalar("data/val_bleu4", scores["BLEU"][3], e)
            tf.summary.scalar("data/val_meteor", scores["METEOR"], e)
            tf.summary.scalar("data/val_rouge", scores["ROUGE"], e)

        # Prepare for next epoch
        best = False
        if val_cider >= best_cider:
            best_cider.assign(val_cider)
            best = True

        # Save model
        ckpt.epoch.assign(e)
        ckpt.write("%s/%s/last" % (args.saved_checkpoint, args.exp_name))

        if best:
            ckpt.write("%s/%s/best" % (args.saved_checkpoint, args.exp_name))



if __name__ == "__main__":
    sys.exit(main())
