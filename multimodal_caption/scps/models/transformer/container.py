import tensorflow as tf
from contextlib import contextmanager


class StatefulModel(tf.keras.Model):
    def __init__(self):
        super(StatefulModel, self).__init__()
        self._is_stateful = False
        self._state_names = []
        self._buffers = dict()
        self._state_defaults = dict()

    def register_state(self, name: str, default: tf.Tensor):
        self._state_names.append(name)
        if default is None:
            self._state_defaults[name] = None
        else:
            self._state_defaults[name] = tf.Variable(default, trainable=False)
        self._buffers[name] = self._state_defaults[name]

    def states(self):
        for name in self._state_names:
            yield self._buffers[name]
        for m in self.layers:
            if isinstance(m, StatefulModel):
                yield from m.states()

    def apply_to_states(self, fn):
        for name in self._state_names:
            self._buffers[name] = fn(self._buffers[name])
        for m in self.layers:
            if isinstance(m, StatefulModel):
                m.apply_to_states(fn)

    def _init_states(self, batch_size: int):
        for name in self._state_names:
            if self._state_defaults[name] is None:
                self._buffers[name] = None
            else:
                self._buffers[name] = self._state_defaults[name]
                self._buffers[name] = tf.expand_dims(self._buffers[name], 0)
                self._buffers[name] = tf.broadcast_to(
                    self._buffers[name], [batch_size,] + list(self._buffers[name].shape[1:]),
                )

    def _reset_states(self):
        for name in self._state_names:
            if self._state_defaults[name] is None:
                self._buffers[name] = None
            else:
                self._buffers[name] = self._state_defaults[name]

    def enable_statefulness(self, batch_size: int):
        for m in self.layers:
            if isinstance(m, StatefulModel):
                m.enable_statefulness(batch_size)
        self._init_states(batch_size)
        self._is_stateful = True

    def disable_statefulness(self):
        for m in self.layers:
            if isinstance(m, StatefulModel):
                m.disable_statefulness()
        self._reset_states()
        self._is_stateful = False

    @contextmanager
    def statefulness(self, batch_size: int):
        self.enable_statefulness(batch_size)
        try:
            yield
        finally:
            self.disable_statefulness()
