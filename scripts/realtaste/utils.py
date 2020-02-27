from keras.layers import Layer


class LastRowLayer(Layer):
    """Extract last time step in a layer with a temporal dimension.

	Implements the same function as
	``layers.core.Lambda(lambda tt: tt[:, -1, :])``.

	This is a workaround to avoid anonymous functions in lambda layers which
	are stored as some kind of (base64-encoded/bitstring/pickled?) string by
	keras, causing the following error during conversion with SNNTB:
		TypeError: ('4wEAAAAAAAAAAQAAAAUAAABTAAAAcxYAAAB8AGQAZACFAmQBZABkAIUCZg
		MZAFMAKQJO6f////+p\nACkB2gJ0dHICAAAAcgIAAAD6Sy9tbnQvMjY0NkJBRjQ0NkJBQzN
		COS9SZXBvc2l0b3JpZXMvTlBQ\nL3Nubl90b29sYm94L3NjcmlwdHMvcmVhbHRhc3RlL3J1
		bi5wedoIPGxhbWJkYT5rAAAA8wAAAAA=\n', None, None)
		is not a callable object

	NOTE: The final part of this string seems to be the base64-encoded path to
	a file - possibly the file in which the layer is defined, or the currently
	running script?

	No noticeable speedup in training between this and a lambda layer.
	"""

    def call(self, x, **kwargs):
        return x[:, -1, :]

    def build(self, input_shape):
        super(LastRowLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        tmp = list(input_shape)
        del tmp[1]
        return tuple(tmp)
