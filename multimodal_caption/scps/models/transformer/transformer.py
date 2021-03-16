import tensorflow as tf
from models.transformer import encoders, decoders
from models.transformer.container import StatefulModel
from models.beam_search import *


class Transformer(StatefulModel):
    def __init__(
        self,
        source_vocab_size,
        target_vocab_size,
        max_len,  # max length of seq, used for postional_encoding
        bos_idx=2,  # check tokenizer.word_index['<bos>']
        eos_idx=3,  # check tokenizer.word_index['<eos>']
        padding_idx=0,  # check tokenizer.word_index['<pad>']
        encoder_layers=2,
        decoder_layers=2,
        d_model=512,
        num_heads=8,
        dff=2048,
        rate=0.1,
    ):
        super(Transformer, self).__init__()

        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.padding_idx = padding_idx

        # inputs are objects detection visual features
        # and image classification labels
        self.visual_encoder = encoders.VisualEncoder(
            padding_idx, encoder_layers, d_model, num_heads, dff, rate
        )

        # word embedding
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)

        # inputs are (English) text
        self.text_encoder = encoders.TextEncoder(
            source_vocab_size,
            max_len,
            padding_idx,
            encoder_layers,
            d_model,
            num_heads,
            dff,
            rate,
        )

        # decoder
        self.decoder = decoders.Decoder(
            target_vocab_size,
            max_len,
            padding_idx,
            decoder_layers,
            d_model,
            num_heads,
            dff,
            rate,
        )

        self.register_state("visual_enc_output", None)
        self.register_state("text_enc_output", None)
        self.register_state("text_enc_mask", None)

    def create_mask(self, x):
        seq_len = tf.shape(x)[1]
        seq = tf.cast(tf.math.equal(x, self.padding_idx), tf.float32)
        padding_mask = seq[:, tf.newaxis, tf.newaxis, :]

        look_ahead_mask = 1 - tf.linalg.band_part(
            tf.ones((seq_len, seq_len)), -1, 0
        )  # (seq_len, seq_len)

        combined_mask = tf.maximum(look_ahead_mask, padding_mask)
        return padding_mask, look_ahead_mask, combined_mask

    def call(self, visual, cls, text, seq, training=False):
        # image features, without masking
        visual_enc_output, visual_enc_mask = self.visual_encoder(
            visual, cls, training
        )  # (batch_size, inp_seq_len, d_model)

        # word embedding for (English) text encoder
        text_enc_mask, _, _ = self.create_mask(text)
        text_embedding = self.embedding(text)

        # text encoding
        text_enc_output, text_enc_mask = self.text_encoder(
            text_embedding, text_enc_mask, training
        )

        # word embedding for decoder
        _, _, seq_combined_mask = self.create_mask(seq)
        seq_embedding = self.embedding(seq)

        # text decoding
        dec_output, _ = self.decoder(
            seq_embedding,
            visual_enc_output,
            text_enc_output,
            visual_enc_mask,
            text_enc_mask,
            seq_combined_mask,
            training,
        )

        return dec_output

    def step(
        self,
        t,
        prev_output,
        visual,
        cls,
        seqs_inp,
        seq,
        mode="teacher_forcing",
        training=False,
        **kwargs
    ):
        it = None
        if mode == "teacher_forcing":
            raise NotImplementedError
        elif mode == "feedback":
            if t == 0:
                self._buffers["visual_enc_output"], _ = self.visual_encoder(
                    visual, cls, training
                )

                # word embedding for text encoder
                enc_mask, _, _ = self.create_mask(seqs_inp)
                embedding = self.embedding(seqs_inp)
                # text encoding
                (
                    self._buffers["text_enc_output"],
                    self._buffers["text_enc_mask"],
                ) = self.text_encoder(embedding, enc_mask, training)

                it = tf.fill([tf.shape(visual)[0], 1], self.bos_idx)
            else:
                it = prev_output

        _, _, it_combined_mask = self.create_mask(it)
        it_embedding = self.embedding(it)

        dec_output, _ = self.decoder(
            it_embedding,
            self._buffers["visual_enc_output"],
            self._buffers["text_enc_output"],
            None,
            self._buffers["text_enc_mask"],
            it_combined_mask,    
            training,
        )

        return dec_output

    def beam_search(
        self,
        visual: tf.Tensor,
        cls: tf.Tensor,
        seqs_inp: tf.Tensor,
        max_len: int,
        eos_idx: int,
        beam_size: int,
        out_size=1,
        return_probs=False,
        training=False,
        **kwargs
    ):
        bs = BeamSearch(self, max_len, eos_idx, beam_size, training)
        return bs.apply(visual, cls, seqs_inp, out_size, return_probs, **kwargs)
