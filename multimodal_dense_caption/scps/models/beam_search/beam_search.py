import tensorflow as tf


class BeamSearch(object):
    def __init__(self, model, max_len: int, eos_idx: int, beam_size: int, training: bool):
        self.model = model
        self.max_len = max_len
        self.eos_idx = eos_idx
        self.beam_size = beam_size
        self.b_s = None
        self.seq_mask = None
        self.seq_logprob = None
        self.outputs = None
        self.log_probs = None
        self.selected_words = None
        self.all_log_probs = None
        self.training = training

    def _expand_state(self, selected_beam: tf.Tensor, cur_beam_size):
        def fn(s):
            shape = tf.shape(s)
            s = tf.gather_nd(
                tf.reshape(s, tf.concat([(self.b_s, cur_beam_size), shape[1:]], axis=0)),
                selected_beam,
            )
            s = tf.reshape(s, tf.concat([(-1,), shape[1:]], axis=0))
            return s

        return fn

    def _expand_visual(
        self, visual: tf.Tensor, cur_beam_size: int, selected_beam: tf.Tensor,
    ):
        visual_shape = tf.shape(visual)
        visual_exp_shape = tf.concat([(self.b_s, cur_beam_size), visual_shape[1:]], axis=0)
        visual_red_shape = tf.concat([(self.b_s * self.beam_size,), visual_shape[1:]], axis=0)
        visual_exp = tf.reshape(visual, visual_exp_shape)
        visual = tf.gather_nd(visual_exp, selected_beam)
        visual = tf.reshape(visual, visual_red_shape)
        return visual

    def apply(self, visual, cls, caps, seqs_inp, out_size=1, return_probs=False, **kwargs):
        self.b_s = tf.shape(visual)[0]
        self.seq_mask = tf.ones((self.b_s, self.beam_size, 1))
        self.seq_logprob = tf.zeros((self.b_s, 1, 1))
        self.log_probs = []
        self.selected_words = None
        if return_probs:
            self.all_log_probs = []

        indices_2d = tf.expand_dims(
            tf.cast(
                tf.broadcast_to(tf.where(tf.ones(self.b_s) != 0), (self.b_s, self.beam_size),),
                dtype=tf.int32,
            ),
            axis=-1,
        )
        indices_3d = tf.cast(
            tf.reshape(
                tf.where(tf.ones((self.b_s, self.beam_size)) != 0), (self.b_s, self.beam_size, -1),
            ),
            dtype=tf.int32,
        )

        outputs = []
        with self.model.statefulness(self.b_s):
            for t in range(self.max_len):
                visual, outputs = self.iter(
                    t, visual, cls, caps, seqs_inp, indices_2d, indices_3d, outputs, return_probs, **kwargs
                )

        # Sort result
        seq_logprob = tf.sort(self.seq_logprob, 1, direction="DESCENDING")
        sort_idxs = tf.argsort(self.seq_logprob, 1, direction="DESCENDING")
        sort_idxs_nd = tf.reshape(
            tf.concat([indices_2d, sort_idxs,], axis=-1,), (self.b_s * self.beam_size, -1),
        )

        outputs = tf.concat(outputs, -1)
        outputs = tf.reshape(tf.gather_nd(outputs, sort_idxs_nd), (self.b_s, self.beam_size, -1))

        log_probs = tf.concat(self.log_probs, -1)
        log_probs = tf.reshape(
            tf.gather_nd(log_probs, sort_idxs_nd,), (self.b_s, self.beam_size, -1)
        )

        if return_probs:
            all_log_probs = tf.concat(self.all_log_probs, 2)
            all_log_probs = tf.gather_nd(all_log_probs, sort_idxs, axis=1, batch_dims=1)

        outputs = outputs[:, :out_size]
        log_probs = log_probs[:, :out_size]
        if out_size == 1:
            outputs = tf.squeeze(outputs, 1)
            log_probs = tf.squeeze(log_probs, 1)
        if return_probs:
            return outputs, log_probs, all_log_probs
        else:
            return outputs, log_probs

    def select(self, candidate_logprob, **kwargs):
        selected_logprob = tf.sort(
            tf.reshape(candidate_logprob, (self.b_s, -1)), -1, direction="DESCENDING"
        )
        selected_idx = tf.argsort(
            tf.reshape(candidate_logprob, (self.b_s, -1)), -1, direction="DESCENDING"
        )
        selected_logprob, selected_idx = (
            selected_logprob[:, : self.beam_size],
            selected_idx[:, : self.beam_size],
        )
        return selected_idx, selected_logprob

    def iter(self, t: int, visual, cls, caps, seqs_inp, indices_2d, indices_3d, outputs, return_probs, **kwargs):
        cur_beam_size = 1 if t == 0 else self.beam_size

        word_logprob = self.model.step(
            t, self.selected_words, visual, cls, caps, seqs_inp, None, mode="feedback", training=self.training, **kwargs
        )
        word_logprob = tf.reshape(word_logprob, (self.b_s, cur_beam_size, -1))
        candidate_logprob = self.seq_logprob + word_logprob

        # Mask sequence if it reaches EOS
        if t > 0:
            mask = tf.expand_dims(
                tf.cast(
                    tf.reshape(self.selected_words, (self.b_s, cur_beam_size)) != self.eos_idx,
                    tf.float32,
                ),
                -1,
            )
            self.seq_mask = self.seq_mask * mask
            word_logprob = word_logprob * tf.broadcast_to(self.seq_mask, tf.shape(word_logprob))
            old_seq_logprob = tf.broadcast_to(self.seq_logprob, tf.shape(candidate_logprob))
            old_seq_logprob = tf.concat(
                [
                    tf.expand_dims(tf.gather(old_seq_logprob, 0, axis=-1), axis=-1),
                    tf.ones(tf.shape(old_seq_logprob) - [0, 0, 1]) * -999,
                ],
                axis=-1,
            )
            candidate_logprob = self.seq_mask * candidate_logprob + old_seq_logprob * (
                1 - self.seq_mask
            )

        selected_idx, selected_logprob = self.select(candidate_logprob, **kwargs)
        selected_beam = tf.cast(selected_idx / tf.shape(candidate_logprob)[-1], dtype=tf.int32)
        selected_words = selected_idx - selected_beam * tf.shape(candidate_logprob)[-1]

        # re-orginize selected beam and words
        selected_beam_nd = tf.reshape(
            tf.concat([indices_2d, tf.expand_dims(selected_beam, axis=-1),], axis=-1,),
            (self.b_s * self.beam_size, -1),
        )

        selected_words_nd = tf.reshape(
            tf.concat([indices_3d, tf.expand_dims(selected_words, axis=-1),], axis=-1,),
            (self.b_s * self.beam_size, -1),
        )

        self.model.apply_to_states(self._expand_state(selected_beam_nd, cur_beam_size))
        #visual = self._expand_visual(visual, cur_beam_size, selected_beam_nd)

        self.seq_logprob = tf.expand_dims(selected_logprob, -1)
        self.seq_mask = tf.reshape(
            tf.gather_nd(self.seq_mask, selected_beam_nd), (self.b_s, self.beam_size, -1),
        )

        outputs = list(
            tf.reshape(tf.gather_nd(o, selected_beam_nd), (self.b_s, self.beam_size, -1))
            for o in outputs
        )
        outputs.append(tf.expand_dims(selected_words, -1))

        if return_probs:
            if t == 0:
                self.all_log_probs.append(
                    tf.expand_dims(
                        tf.broadcast_to(word_logprob, (self.b_s, self.beam_size, -1)), axis=2,
                    ),
                )
            else:
                self.all_log_probs.append(tf.expand_dims(word_logprob, axis=2))

        this_word_logprob = tf.reshape(
            tf.gather_nd(word_logprob, selected_beam_nd), (self.b_s, self.beam_size, -1)
        )
        this_word_logprob = tf.reshape(
            tf.gather_nd(this_word_logprob, selected_words_nd), (self.b_s, self.beam_size, -1),
        )
        self.log_probs = list(
            tf.reshape(tf.gather_nd(o, selected_beam_nd), (self.b_s, self.beam_size, -1))
            for o in self.log_probs
        )
        self.log_probs.append(this_word_logprob)
        self.selected_words = tf.reshape(selected_words, (-1, 1))

        return visual, outputs
