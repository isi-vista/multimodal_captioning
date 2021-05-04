import tensorflow as tf
from models.transformer.attention import MultiHeadAttention
from models.transformer.utils import point_wise_feed_forward_network
from models.transformer.utils import (
    point_wise_feed_forward_network,
    positional_encoding,
)


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, q, k, v, training, mask):
        attn_output, _ = self.mha(v, k, q, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(q + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(
            out1 + ffn_output
        )  # (batch_size, input_seq_len, d_model)

        return out2


class TextEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        target_vocab_size,
        max_len,
        padding_idx=0,
        num_layers=3,
        d_model=512,
        num_heads=8,
        dff=2048,
        rate=0.1,
    ):
        super(TextEncoder, self).__init__()

        self.vocab_size = target_vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.padding_idx = padding_idx

        # self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_len, d_model)
        self.dropout = tf.keras.layers.Dropout(rate)

        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]

    def create_mask(self, x):
        seq = tf.cast(tf.math.equal(x, self.padding_idx), tf.float32)

        return seq[:, tf.newaxis, tf.newaxis, :]

    def call(self, x, mask, training=False):
        # mask = self.create_mask(x)

        ## adding embedding and position encoding.
        seq_len = tf.shape(x)[1]
        # x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, x, x, training, mask)

        return x, mask  # (batch_size, input_seq_len, d_model)


class VisualEncoder(tf.keras.layers.Layer):
    def __init__(
        self, padding_idx=0, num_layers=3, d_model=512, num_heads=8, dff=2048, rate=0.1
    ):
        super(VisualEncoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.padding_idx = padding_idx

        # Since you have already extracted the features and dumped it using pickle
        # This encoder passes those features through a Fully connected layer
        self.embedding = tf.keras.layers.Embedding(9000, d_model)
        self.dense = tf.keras.layers.Dense(d_model, activation="relu")

        self.enc_layers_x = [
            EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]
        self.enc_layers_y = [
            EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]
        self.enc_layers_xy = [
            EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(rate)

    def create_mask(self, x):
        return None

    def call(self, x, y, training=False):
        """ x: input visual features
            y: input class labels
        """
        mask = self.create_mask(x)

        x = self.dense(x)
        x = self.dropout(x, training=training)
        y = self.embedding(y)

        for i in range(self.num_layers):
            x = self.enc_layers_x[i](x, x, x, training, mask)

        for i in range(self.num_layers):
            y = self.enc_layers_y[i](y, y, y, training, mask)

        for i in range(self.num_layers):
            x = self.enc_layers_xy[i](x, y, y, training, mask)

        return x, mask  # (batch_size, input_seq_len, d_model)
