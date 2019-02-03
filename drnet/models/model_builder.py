"""
Copyright (C) 2019  Patrick Schwab, ETH Zurich

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
 of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.regularizers import L1L2
from keras.optimizers import Adam, RMSprop
from keras.layers import Input, Dense, Dropout, Lambda, LSTM, Reshape, concatenate, Activation, Bidirectional, \
    TimeDistributed, Conv1D, MaxPooling1D, Flatten, GlobalAveragePooling1D, dot, multiply
from keras.layers.normalization import BatchNormalization
from drnet.models.distributions import wasserstein, safe_sqrt
from drnet.models.per_sample_dropout import PerSampleDropout


class ModelBuilder(object):
    @staticmethod
    def compile_model(model, learning_rate, optimizer="rmsprop", loss_weights=list([1.0]),
                      main_loss="mse", extra_loss=None, metrics={}):

        losses = main_loss

        if loss_weights is not None:
            losses = [losses] * len(loss_weights)

        if extra_loss is not None:
            if isinstance(extra_loss, list):
                for i in range(1, 1+len(extra_loss)):
                    losses[i] = extra_loss[i-1]
            else:
                losses[1] = extra_loss

        if optimizer == "rmsprop":
            opt = RMSprop(lr=learning_rate)
        else:
            opt = Adam(lr=learning_rate)

        model.compile(loss=losses if len(losses) > 1 else losses[0],
                      loss_weights=loss_weights,
                      optimizer=opt,
                      metrics=metrics)
        return model

    @staticmethod
    def build_mlp(last_layer, p_dropout=0.0, num_layers=1, with_bn=True, dim=None, l2_weight=0.0,
                  last_activity_regulariser=None, propensity_dropout=None, normalize=False, activation="linear"):
        if dim is None:
            dim = K.int_shape(last_layer)[-1]

        for i in range(num_layers):
            last_layer = Dense(dim,
                               kernel_initializer="he_normal",
                               bias_initializer="zeros",
                               kernel_regularizer=L1L2(l2=l2_weight),
                               bias_regularizer=L1L2(l2=l2_weight),
                               use_bias=not with_bn,
                               activity_regularizer=last_activity_regulariser if i == num_layers-1 else None,
                               activation=activation)\
                (last_layer)

            if with_bn:
                last_layer = BatchNormalization(gamma_regularizer=L1L2(l2=l2_weight),
                                                beta_regularizer=L1L2(l2=l2_weight))(last_layer)

            last_layer = Dropout(p_dropout)(last_layer)
            if propensity_dropout is not None:
                last_layer = PerSampleDropout(propensity_dropout)(last_layer)

        if normalize:
            last_layer = Lambda(lambda x: x / (safe_sqrt(tf.reduce_sum(tf.square(x),
                                                                       axis=1,
                                                                       keep_dims=True)) + K.epsilon()))(last_layer)

        if last_activity_regulariser is not None:
            identity_layer = Lambda(lambda x: x)
            identity_layer.activity_regularizer = last_activity_regulariser
            last_layer = identity_layer(last_layer)

        return last_layer

    @staticmethod
    def build_multi_stream(input_dim, output_dim, num_units=128, dropout=0.0, l2_weight=0.0, learning_rate=0.0001, num_layers=2,
                           num_treatments=2, p_ipm=0.5, imbalance_loss_weight=1.0, with_bn=False, with_propensity_dropout=True,
                           normalize=True,
                           **kwargs):
        treatment_input = Input(shape=(1,), dtype="int32")
        all_inputs, topmost_hidden_layers = [], [Lambda(lambda x: K.cast(x, "float32"))(treatment_input)]
        for input_stream in input_dim:
            input_layer = Input(input_stream)

            last_layer = input_layer
            last_layer = Reshape((input_stream[0], 1))(last_layer)
            cur_num_units = num_units
            for _ in range(4):
                last_layer = Conv1D(cur_num_units, kernel_size=3,
                                    padding="same",
                                    kernel_initializer="he_normal",
                                    activation="elu")(last_layer)
                last_layer = MaxPooling1D(2)(last_layer)
                cur_num_units += 8
            last_layer = GlobalAveragePooling1D()(last_layer)

            all_inputs.append(input_layer)
            topmost_hidden_layers.append(last_layer)

        last_layer = concatenate(topmost_hidden_layers)
        last_layer = Dense(1024, activation="elu")(last_layer)
        last_layer = Dense(output_dim, activation="linear")(last_layer)

        all_inputs.append(treatment_input)
        model = Model(inputs=all_inputs,
                      outputs=last_layer)
        model.summary()

        main_model = ModelBuilder.compile_model(model, learning_rate,
                                                loss_weights=[1.0],
                                                main_loss="mse")

        return main_model

    @staticmethod
    def build_simple(input_dim, output_dim, num_units=128, dropout=0.0, l2_weight=0.0, learning_rate=0.0001, num_layers=2,
                     num_treatments=2, p_ipm=0.5, imbalance_loss_weight=1.0, with_bn=False, with_propensity_dropout=True,
                     normalize=True, with_exposure=False,
                     **kwargs):
        covariate_input = Input(shape=(input_dim,))
        treatment_input = Input(shape=(1,), dtype="int32")
        if with_exposure:
            exposure_input = Input(shape=(1,), dtype="float32")
            treatment_num = Lambda(lambda x: K.cast(x, "float32"))(treatment_input)
            inputs = [covariate_input, treatment_input, exposure_input]
            to_concatenate = [covariate_input, treatment_num, exposure_input]
        else:
            treatment_num = Lambda(lambda x: K.cast(x, "float32"))(treatment_input)
            inputs = [covariate_input, treatment_input]
            to_concatenate = [covariate_input, treatment_num]

        model_outputs, loss_weights = [], []
        propensity_dropout = None
        regulariser = None

        last_layer = concatenate(to_concatenate, axis=-1)
        last_layer = ModelBuilder.build_mlp(last_layer,
                                            dim=num_units,
                                            p_dropout=dropout,
                                            num_layers=num_layers,
                                            with_bn=with_bn,
                                            l2_weight=l2_weight,
                                            propensity_dropout=propensity_dropout,
                                            normalize=normalize,
                                            last_activity_regulariser=regulariser,
                                            activation="elu")

        all_outputs = Dense(output_dim, activation="linear", name="head")(last_layer)
        output = all_outputs

        model_outputs.append(output)
        loss_weights.append(1)

        model = Model(inputs=inputs,
                      outputs=model_outputs)
        model.summary()

        main_model = ModelBuilder.compile_model(model, learning_rate,
                                                loss_weights=loss_weights,
                                                main_loss="mse")

        return main_model

    @staticmethod
    def build_tarnet(input_dim, output_dim, num_units=128, dropout=0.0, l2_weight=0.0, learning_rate=0.0001, num_layers=2,
                     num_treatments=2, p_ipm=0.5, imbalance_loss_weight=1.0, with_bn=False, with_propensity_dropout=True,
                     normalize=True, with_exposure=False, num_exposure_strata=5, with_exposure_strata=True,
                     with_multiple_exposure_inputs=True,
                     **kwargs):
        rnaseq_input = Input(shape=(input_dim,))
        treatment_input = Input(shape=(1,), dtype="int32")

        if with_exposure:
            exposure_input = Input(shape=(1,), dtype="float32")
        model_outputs, loss_weights = [], []

        if with_exposure_strata:
            last_layer = rnaseq_input
        else:
            last_layer = concatenate([rnaseq_input, exposure_input], axis=-1)

        if with_propensity_dropout:
            dropout = 0
            propensity_output = ModelBuilder.build_mlp(last_layer,
                                                       dim=num_units,
                                                       p_dropout=dropout,
                                                       num_layers=num_layers,
                                                       with_bn=with_bn,
                                                       l2_weight=l2_weight,
                                                       activation="elu")
            propensity_output = Dense(num_treatments, activation="softmax", name="propensity")(propensity_output)
            model_outputs.append(propensity_output)
            loss_weights.append(1)
            gamma = 0.5
            propensity_dropout = Lambda(lambda x: tf.stop_gradient(x))(propensity_output)

            def get_treatment_propensities(x):
                cat_idx = tf.stack([tf.range(0, tf.shape(x[0])[0]), K.squeeze(tf.cast(x[1], "int32"), axis=-1)],
                                   axis=1)
                return tf.gather_nd(x[0], cat_idx)

            propensity_dropout = Lambda(get_treatment_propensities)([propensity_dropout, treatment_input])
            propensity_dropout = Lambda(lambda x: 1. - gamma -
                                                  1./2. * (-x * tf.log(x + K.epsilon()) -
                                                           (1 - x)*tf.log(1 - x + K.epsilon())))\
                (propensity_dropout)
        else:
            propensity_dropout = None

        regulariser = None
        if imbalance_loss_weight != 0.0:

            def wasserstein_distance_regulariser(x):
                return imbalance_loss_weight*wasserstein(x, treatment_input, p_ipm,
                                                         num_treatments=num_treatments)

            regulariser = wasserstein_distance_regulariser

        # Build shared representation.
        last_layer = ModelBuilder.build_mlp(last_layer,
                                            dim=num_units,
                                            p_dropout=dropout,
                                            num_layers=num_layers,
                                            with_bn=with_bn,
                                            l2_weight=l2_weight,
                                            propensity_dropout=propensity_dropout,
                                            normalize=normalize and not with_bn,
                                            last_activity_regulariser=regulariser,
                                            activation="elu")

        last_layer_h = last_layer

        all_indices, outputs = [], []
        for i in range(num_treatments):

            def get_indices_equal_to(x):
                return tf.reshape(tf.to_int32(tf.where(tf.equal(tf.reshape(x, (-1,)), i))), (-1,))

            indices = Lambda(get_indices_equal_to)(treatment_input)

            # indices = Lambda(lambda x: tf.Print(x, [x], "indices(" + str(i) + ")=", summarize=8))(indices)

            current_last_layer_h = Lambda(lambda x: tf.gather(x, indices))(last_layer_h)

            if with_propensity_dropout:
                current_propensity_dropout = Lambda(lambda x: tf.gather(x, indices))(propensity_dropout)
            else:
                current_propensity_dropout = None

            last_layer = ModelBuilder.build_mlp(current_last_layer_h,
                                                dim=num_units,
                                                p_dropout=dropout,
                                                num_layers=num_layers,
                                                with_bn=with_bn,
                                                propensity_dropout=current_propensity_dropout,
                                                normalize=normalize and not with_bn,
                                                l2_weight=l2_weight,
                                                activation="elu")

            if with_exposure and with_exposure_strata:
                current_exposure = Lambda(lambda x: tf.gather(x, indices))(exposure_input)

                step_size = 1.0 / num_exposure_strata
                for bin_idx in range(num_exposure_strata):
                    def get_indices_between(x):
                        lower_limit = step_size*bin_idx
                        upper_limit = step_size*(bin_idx+1)
                        upper_indices = tf.greater_equal(tf.reshape(x, (-1,)), lower_limit)
                        if bin_idx == num_exposure_strata - 1:
                            lower_indices = tf.less_equal(tf.reshape(x, (-1,)), upper_limit)
                        else:
                            lower_indices = tf.less(tf.reshape(x, (-1,)), upper_limit)
                        bounded = tf.where(tf.logical_and(upper_indices, lower_indices))
                        return tf.reshape(tf.to_int32(bounded), (-1,))

                    sub_indices = Lambda(get_indices_between)(current_exposure)

                    sub_last_layer = Lambda(lambda x: tf.gather(x, sub_indices))(last_layer)
                    sub_exposure = Lambda(lambda x: tf.gather(x, sub_indices))(current_exposure)
                    sub_exposure = Lambda(lambda x: (x - 0.5) * 2.)(sub_exposure)

                    if with_propensity_dropout:
                        sub_propensity_dropout = Lambda(lambda x: tf.gather(x,
                                                                            sub_indices))(current_propensity_dropout)
                    else:
                        sub_propensity_dropout = None

                    for layer_idx in range(num_layers):
                        if with_multiple_exposure_inputs or layer_idx == 1:
                            exposure_layer = concatenate([sub_last_layer, sub_exposure])
                        else:
                            exposure_layer = sub_last_layer
                        sub_last_layer = ModelBuilder.build_mlp(exposure_layer,
                                                                dim=num_units,
                                                                p_dropout=0,
                                                                num_layers=1,
                                                                with_bn=with_bn,
                                                                propensity_dropout=sub_propensity_dropout,
                                                                l2_weight=l2_weight,
                                                                normalize=normalize and
                                                                          not with_bn and
                                                                          layer_idx == num_layers - 1,
                                                                activation="elu")

                    if with_multiple_exposure_inputs:
                        sub_last_layer = concatenate([sub_exposure, sub_last_layer])

                    output = Dense(output_dim, activation="linear",
                                   name="head_" + str(i) + str(bin_idx),
                                   kernel_initializer="random_uniform",
                                   bias_initializer="zeros")(sub_last_layer)

                    sub_indices_in_indices = Lambda(lambda x: tf.gather(x, sub_indices))(indices)
                    all_indices.append(sub_indices_in_indices)
                    outputs.append(output)
            else:
                output = Dense(output_dim, activation="linear", name="head_" + str(i))(last_layer)

                all_indices.append(indices)
                outputs.append(output)

        def do_dynamic_stitch(x):
            num_tensors = len(x)

            # x = map(lambda xi, i: K.print_tensor(xi, "stitch(" + str(i) + ")="), x, range(len(x)))

            data_indices = map(tf.to_int32, x[:num_tensors/2])
            data = map(tf.to_float, x[num_tensors/2:])

            stitched = tf.dynamic_stitch(data_indices, data)
            return stitched

        output = Lambda(do_dynamic_stitch, name="dynamic_stitch")(all_indices + outputs)
        model_outputs.append(output)
        loss_weights.append(1)

        inputs = [rnaseq_input, treatment_input]
        if with_exposure:
            inputs += [exposure_input]
        model = Model(inputs=inputs,
                      outputs=model_outputs)
        model.summary()

        main_model = ModelBuilder.compile_model(model, learning_rate,
                                                loss_weights=loss_weights,
                                                main_loss="mse")

        return main_model

    @staticmethod
    def set_layers_trainable(model, trainable):
        model.trainable = trainable
        for layer in model.layers:
            layer.trainable = trainable
        return model
