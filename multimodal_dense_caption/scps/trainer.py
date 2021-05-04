import tensorflow as tf
import numpy as np
import itertools
import multiprocessing
import evaluation
from evaluation import PTBTokenizer, Cider, utils
from tqdm import tqdm


def xe_loss(real, pred, scores):
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )
    loss_ = loss_obj(real, pred)
    loss_ = tf.multiply(loss_, tf.expand_dims(scores, axis=-1))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    mask = tf.cast(mask, dtype=loss_.dtype)

    loss_ *= mask

    return tf.reduce_mean(loss_)


def evaluate_metrics(model, dataloader, tokenizer, beam_size, epoch):
    gen = {}
    gts = {}

    # setup decoding graph
    #@tf.function
    def beam_search(
        model, visual, cls, dense_caps, seqs_inp, max_len, eos_idx, beam_size, out_size
    ):
        out, log_probs = model.beam_search(
            visual,
            cls,
            dense_caps, 
            seqs_inp,
            max_len,
            eos_idx,
            beam_size,
            out_size,
        )
        return out, log_probs

    with tqdm(
        desc="Epoch %d - evaluation" % epoch, unit="it", total=dataloader["data_steps"],
    ) as pbar:
        for it, (visual, cls, dense_caps, seqs_inp, seqs_gts, caps_gts, _) in enumerate(
            dataloader["data"]
        ):
            out, _ = beam_search(
                model,
                visual,
                cls,
                list(np.transpose(dense_caps, (1, 0, 2))),
                seqs_inp,
                50,
                model.eos_idx,
                beam_size,
                out_size=1,
            )
            caps_gen = tokenizer.sequences_to_texts(tf.squeeze(out).numpy())
            caps_gen = utils.clean_text(caps_gen, "<eos>")

            for i, (gts_i, gen_i) in enumerate(zip(caps_gts.numpy(), caps_gen)):
                gen["%d_%d" % (it, i)] = [gen_i]
                gts["%d_%d" % (it, i)] = [gts_i.decode("utf-8")]

            pbar.update()

    score_gts = evaluation.PTBTokenizer.tokenize(gts)
    score_gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(score_gts, score_gen)

    return scores, gts, gen


def evaluate_loss(model, dataloader, epoch):
    running_loss = 0.0
    loss_fn = xe_loss

    @tf.function
    def evaluate(visual, cls, seqs_inp, caps_gts):
        out = model(visual, cls, seqs_inp, caps_gts[:, :-1], True)
        loss = loss_fn(caps_gts[:, 1:], out)

        return loss

    with tqdm(
        desc="Epoch %d - validation" % epoch, unit="it", total=dataloader["data_steps"]
    ) as pbar:
        for it, (visual, cls, seqs_inp, seqs_gts, caps_gts) in enumerate(
            dataloader["data"]
        ):
            loss = evaluate(visual, cls, seqs_inp, seqs_gts)
            running_loss += loss
            pbar.set_postfix(loss=running_loss.numpy() / (it + 1))
            pbar.update()

    return running_loss / dataloader["data_steps"]


def train_xe(model, dataloader, optim, epoch):
    running_loss = 0.0
    loss_fn = xe_loss
    train_loss = tf.keras.metrics.Mean(name="train_loss")

    # @tf.function
    def train(visual, cls, dense_caps, seqs_inp, seqs_gts, scores):
        with tf.GradientTape() as tape:
            out = model(visual, cls, dense_caps, seqs_inp, seqs_gts[:, :-1], True)
            loss = loss_fn(seqs_gts[:, 1:], out, scores)

        gradients = tape.gradient(loss, model.trainable_variables)
        optim.apply_gradients(zip(gradients, model.trainable_variables))
        return train_loss(loss)

    with tqdm(
        desc="Epoch %d - train-xe" % epoch, unit="it", total=dataloader["data_steps"]
    ) as pbar:
        for (
            it,
            (visual, cls, dense_caps, seqs_inp, seqs_gts, caps_gts, scores),
        ) in enumerate(dataloader["data"]):
            loss = train(
                visual,
                cls,
                list(np.transpose(dense_caps, (1, 0, 2))),
                seqs_inp,
                seqs_gts,
                scores,
            )
            running_loss += loss
            pbar.set_postfix(loss=loss.numpy())
            pbar.update()

    return running_loss / dataloader["data_steps"]
