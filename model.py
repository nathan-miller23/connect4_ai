from ray.rllib.models.tf.tf_modelv2 import TFModelV2
import tensorflow as tf

class RllibPPOModel(TFModelV2):
    """
    Model that will map environment states to action probabilities. Will be shared across agents
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        super(RllibPPOModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        # params we got to pass in from the call to "run"
        custom_params = model_config["custom_model_config"]


        ## Parse custom network params
        num_hidden_layers = custom_params["NUM_HIDDEN_LAYERS"]
        size_hidden_layers = custom_params["SIZE_HIDDEN_LAYERS"]
        num_filters = custom_params["NUM_FILTERS"]
        num_convs = custom_params["NUM_CONV_LAYERS"]
        d2rl = custom_params["D2RL"]
        assert type(d2rl) == bool

        ## Create graph of custom network. It will under a shared tf scope such that all agents
        ## use the same model
        self.inputs = tf.keras.Input(shape=obs_space.shape, name="observations")
        out = self.inputs

        # Apply remaining conv layers, if any
        for i in range(0, num_convs):
            padding = "same"
            out = tf.keras.layers.Conv2D(
                filters=num_filters,
                kernel_size=[3, 3],
                padding=padding,
                activation=tf.nn.leaky_relu,
                name="conv_{}".format(i)
            )(out)
        
        # Apply dense hidden layers, if any
        conv_out = tf.keras.layers.Flatten()(out)
        out = conv_out
        for i in range(num_hidden_layers):
            out = tf.keras.layers.Dense(size_hidden_layers)(out)
            out = tf.keras.layers.LeakyReLU()(out)

        # Linear last layer for action distribution logits
        layer_out = tf.keras.layers.Dense(self.num_outputs)(out)

        # Linear last layer for value function branch of model
        value_out = tf.keras.layers.Dense(1)(out)

        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])


    def forward(self, input_dict, state=None, seq_lens=None):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])