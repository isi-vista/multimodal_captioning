import tensorflow as tf
from models.transformer.container import StatefulModel
from models.transformer.attention import MultiHeadAttention
from models.transformer.utils import (
    point_wise_feed_forward_network,
    positional_encoding,
)


class DecoderLayer(StatefulModel):
    def __init__(self, d_model=512, num_heads=8, dff=2048, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.self_att = MultiHeadAttention(d_model, num_heads, can_be_stateful=True)
        self.visual_enc_att = MultiHeadAttention(
            d_model, num_heads, can_be_stateful=False
        )
        self.text_enc_att = MultiHeadAttention(
            d_model, num_heads, can_be_stateful=False
        )

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm4 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
        self.dropout4 = tf.keras.layers.Dropout(rate)

        self.visual_alpha = tf.keras.layers.Dense(d_model)
        self.text_alpha = tf.keras.layers.Dense(d_model) 

        self.fn = tf.keras.layers.Dense(d_model)

    def call(
        self,
        x,
        visual_enc_output,
        text_enc_output,
        mask_visual_enc_att,
        mask_text_enc_att,
        mask_self_att,
        training,
    ):
        # enc_output.shape == (batch_size, input_seq_len, d_model)
        self_attn_output, self_attn_weights = self.self_att(
            x, x, x, mask_self_att
        )  # (batch_size, target_seq_len, d_model)
        self_attn_output = self.dropout1(self_attn_output, training=training)
        self_attn_output = self.layernorm1(self_attn_output + x)

        # attend to visual
        visual_enc_attn_output, visual_enc_attn_weights = self.visual_enc_att(
            visual_enc_output, visual_enc_output, self_attn_output, mask_visual_enc_att
        )  # (batch_size, target_seq_len, d_model)
        visual_enc_attn_output = self.dropout2(
            visual_enc_attn_output, training=training
        )
        visual_enc_attn_output = self.layernorm2(
            visual_enc_attn_output + self_attn_output
        )  # (batch_size, target_seq_len, d_model)
        visual_alpha = tf.math.sigmoid(
            self.visual_alpha(
                tf.concat([visual_enc_attn_output, self_attn_output], axis=-1)
            )
        )
        visual_enc_attn_output = tf.multiply(visual_alpha, visual_enc_attn_output)

        # attend to text
        text_enc_attn_output, text_enc_attn_weights = self.text_enc_att(
            text_enc_output, text_enc_output, self_attn_output, mask_text_enc_att
        )
        text_enc_attn_output = self.dropout3(text_enc_attn_output, training=training)
        text_enc_attn_output = self.layernorm3(
            text_enc_attn_output + self_attn_output
        )  # (batch_size, target_seq_len, d_model)
        text_alpha = tf.math.sigmoid(
            self.text_alpha(
                tf.concat([text_enc_attn_output, self_attn_output], axis=-1)
            )
        )
        text_enc_attn_output = tf.multiply(text_alpha, text_enc_attn_output)

        merged_enc_attn_output = tf.concat(
            [visual_enc_attn_output, text_enc_attn_output], axis=-1
        )
        merged_enc_attn_output = self.fn(merged_enc_attn_output)

        ffn_output = self.ffn(
            merged_enc_attn_output
        )  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout4(ffn_output, training=training)
        output = self.layernorm4(
            ffn_output + merged_enc_attn_output
        )  # (batch_size, target_seq_len, d_model)

        return output, self_attn_weights, visual_enc_attn_weights, text_enc_attn_weights


class Decoder(StatefulModel):
    def __init__(
        self,
        target_vocab_size,
        max_len,  # used for maximum_position_encoding
        padding_idx=0,
        num_layers=6,
        d_model=512,
        num_heads=8,
        dff=2048,
        rate=0.1,
    ):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.padding_idx = padding_idx

        # self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_len, d_model)
        self.dropout = tf.keras.layers.Dropout(rate)

        self.dec_layers = [
            DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]

        self.final_layer = tf.keras.layers.Dense(target_vocab_size,)

        # state for beam search decoding
        self.register_state("running_mask_self_attention", tf.zeros((1, 1, 0)))
        self.register_state("running_seq", tf.zeros((1,)))

    def create_mask(self, x):
        seq_len = tf.shape(x)[1]
        seq = tf.cast(tf.math.equal(x, self.padding_idx), tf.float32)

        mask = 1 - tf.linalg.band_part(
            tf.ones((seq_len, seq_len)), -1, 0
        )  # (seq_len, seq_len)

        return tf.maximum(mask, seq[:, tf.newaxis, tf.newaxis, :])

    def call(
        self,
        x,
        visual_enc_output,
        text_enc_output,
        visual_encoder_mask=None,
        text_encoder_mask=None,
        seq_decoder_mask=None,
        training=False,
    ):
        # mask_self_attention = self.create_mask(x)
        mask_self_attention = seq_decoder_mask

        if self._is_stateful:
            self._buffers["running_mask_self_attention"] = tf.concat(
                [self._buffers["running_mask_self_attention"], mask_self_attention], -1
            )
            mask_self_attention = self._buffers["running_mask_self_attention"]

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        # x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        if self._is_stateful:
            i = tf.cast(self._buffers["running_seq"][0][0], dtype=tf.int32)
            x += self.pos_encoding[:, i : i + seq_len, :]
            self._buffers["running_seq"] += 1
        else:
            x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2, block3 = self.dec_layers[i](
                x,
                visual_enc_output,
                text_enc_output,
                visual_encoder_mask,
                text_encoder_mask,
                mask_self_attention,
                training,
            )

            attention_weights["decoder_layer{}_block1".format(i + 1)] = block1
            attention_weights["decoder_layer{}_block2".format(i + 1)] = block2
            attention_weights["decoder_layer{}_block3".format(i + 1)] = block3

        final_output = tf.nn.log_softmax(self.final_layer(x), -1)

        # x.shape == (batch_size, target_seq_len, d_model)
        return final_output, attention_weights
