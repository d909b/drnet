"""
Copyright (C) 2019  anonymised author, anonymised institution

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
from ..cfr.util import get_nonlinearity_by_name, build_mlp


class GANITEBuilder(object):
    @staticmethod
    def build(input_dim, output_dim, num_units=128, dropout=0.0, l2_weight=0.0, learning_rate=0.0001, num_layers=2,
              num_treatments=2, with_bn=False, nonlinearity="elu", initializer=tf.variance_scaling_initializer(),
              alpha=1.0, beta=1.0, with_exposure=False):
        x = tf.placeholder("float", shape=[None, input_dim], name='x')
        t = tf.placeholder("float", shape=[None, 1], name='t')
        y_f = tf.placeholder("float", shape=[None, 1], name='y_f')
        y_full = tf.placeholder("float", shape=[None, num_treatments], name='y_full')

        if with_exposure:
            ts = tf.placeholder("float", shape=[None, 1], name='ts')
        else:
            ts = None

        y_pred_cf, propensity_scores, z_g = GANITEBuilder.build_counterfactual_block(input_dim, x, t, y_f, ts,
                                                                                     num_units, dropout, l2_weight,
                                                                                     learning_rate, num_layers,
                                                                                     num_treatments, with_bn,
                                                                                     nonlinearity, initializer)

        y_pred_ite, d_ite_pred, d_ite_true, z_i = GANITEBuilder.build_ite_block(input_dim, x, t, y_f, y_full, ts,
                                                                                num_units, dropout, l2_weight,
                                                                                learning_rate, num_layers,
                                                                                num_treatments, with_bn,
                                                                                nonlinearity, initializer)

        # Build losses and optimizers.
        t_one_hot = tf.one_hot(tf.cast(t, "int32"), num_treatments)

        propensity_loss_cf = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=propensity_scores,
                                                                                    labels=t_one_hot))

        batch_size = tf.shape(y_pred_cf)[0]
        indices = tf.stack([tf.range(batch_size), tf.cast(t, "int32")[:, 0]], axis=-1)
        y_f_pred = tf.gather_nd(y_pred_cf, indices)

        y_f_i = y_f  # tf.Print(y_f, [y_f[:, 0]], message="y_f=", summarize=8)
        y_f_pred_i = y_f_pred  # tf.Print(y_f_pred, [y_f_pred], message="y_f_pred=", summarize=8)

        supervised_loss_cf = tf.sqrt(tf.reduce_mean(tf.squared_difference(y_f_i[:, 0], y_f_pred_i)))

        cf_discriminator_loss = propensity_loss_cf
        cf_generator_loss = -propensity_loss_cf + alpha * supervised_loss_cf

        # D_ITE goal: 0 when True, 1 when Pred
        ite_loss = tf.reduce_mean(tf.log(d_ite_true)) + tf.reduce_mean(tf.log(1 - d_ite_pred))

        y_full_i = y_full  # tf.Print(y_full, [y_full], message="y_full=", summarize=8)
        y_pred_ite_i = y_pred_ite  # tf.Print(y_pred_ite, [y_pred_ite], message="y_pred_ite=", summarize=8)
        supervised_loss_ite = tf.sqrt(tf.reduce_mean(tf.squared_difference(y_full_i, y_pred_ite_i)))

        ite_discriminator_loss = -ite_loss
        ite_generator_loss = ite_loss + beta * supervised_loss_ite
        return cf_generator_loss, cf_discriminator_loss, ite_generator_loss, ite_discriminator_loss, \
               x, t, y_f, y_full, ts, y_pred_cf, y_pred_ite, z_g, z_i

    @staticmethod
    def build_tarnet(mlp_input, t, input_dim, num_layers, num_units, dropout, num_treatments, nonlinearity):
        initializer = tf.variance_scaling_initializer()
        x = build_mlp(mlp_input, num_layers, num_units, dropout, nonlinearity)[0]

        all_indices, outputs = [], []
        for i in range(num_treatments):
            indices = tf.reshape(tf.to_int32(tf.where(tf.equal(tf.reshape(t, (-1,)), i))), (-1,))
            current_last_layer_h = tf.gather(x, indices)

            last_layer = build_mlp(current_last_layer_h, num_layers, num_units, dropout, nonlinearity)[0]

            output = tf.layers.dense(last_layer, units=num_treatments, use_bias=True,
                                     bias_initializer=initializer)

            all_indices.append(indices)
            outputs.append(output)
        return tf.concat(outputs, axis=-1), all_indices

    @staticmethod
    def build_counterfactual_block(input_dim, x, t, y_f, ts, num_units=128, dropout=0.0, l2_weight=0.0,
                                   learning_rate=0.0001, num_layers=2,
                                   num_treatments=2, with_bn=False, nonlinearity="elu",
                                   initializer=tf.variance_scaling_initializer()):

        y_pred, z_g = GANITEBuilder.build_counterfactual_generator(input_dim, x, t, y_f, ts, num_units,
                                                                   dropout, l2_weight, learning_rate,
                                                                   num_layers, num_treatments, with_bn,
                                                                   nonlinearity,
                                                                   initializer)

        propensity_scores = GANITEBuilder.build_counterfactual_discriminator(input_dim, x, t, y_pred, ts, num_units,
                                                                             dropout, l2_weight, learning_rate,
                                                                             num_layers, num_treatments, with_bn,
                                                                             nonlinearity,
                                                                             initializer)
        return y_pred, propensity_scores, z_g

    @staticmethod
    def build_counterfactual_generator(input_dim, x, t, y_f, ts, num_units=128, dropout=0.0, l2_weight=0.0,
                                       learning_rate=0.0001, num_layers=2,
                                       num_treatments=2, with_bn=False, nonlinearity="elu",
                                       initializer=tf.variance_scaling_initializer()):
        nonlinearity = get_nonlinearity_by_name(nonlinearity)
        with tf.variable_scope("g_cf",
                               initializer=initializer):
            z_g = tf.placeholder("float", shape=[None, num_treatments-1], name='z_g')

            mlp_input = tf.concat([x, y_f, t, z_g], axis=-1)

            if ts is not None:
                mlp_input = tf.concat([mlp_input, ts], axis=-1)

            x = build_mlp(mlp_input, num_layers, num_units, dropout, nonlinearity)[0]
            y = tf.layers.dense(x, units=num_treatments, use_bias=True,
                                bias_initializer=initializer)
            return y, z_g

    @staticmethod
    def build_counterfactual_discriminator(input_dim, x, t, y_pred, ts, num_units=128, dropout=0.0, l2_weight=0.0,
                                           learning_rate=0.0001, num_layers=2,
                                           num_treatments=2, with_bn=False, nonlinearity="elu",
                                           initializer=tf.variance_scaling_initializer(),
                                           reuse=False):
        nonlinearity = get_nonlinearity_by_name(nonlinearity)
        with tf.variable_scope("d_cf",
                               reuse=reuse,
                               initializer=initializer):
            mlp_input = tf.concat([x, y_pred], axis=-1)
            x = build_mlp(mlp_input, num_layers, num_units, dropout, nonlinearity)[0]
            propensity_scores = tf.layers.dense(x, units=num_treatments, use_bias=True,
                                                bias_initializer=initializer)
            return propensity_scores

    @staticmethod
    def build_ite_block(input_dim, x, t, y_f, y_full, ts, num_units=128, dropout=0.0, l2_weight=0.0,
                        learning_rate=0.0001, num_layers=2,
                        num_treatments=2, with_bn=False, nonlinearity="elu",
                        initializer=tf.variance_scaling_initializer()):
        y_pred_ite, z_i = GANITEBuilder.build_ite_generator(input_dim, x, t, y_f, ts, num_units,
                                                            dropout, l2_weight, learning_rate,
                                                            num_layers, num_treatments, with_bn,
                                                            nonlinearity, initializer)

        d_ite_pred = GANITEBuilder.build_ite_discriminator(input_dim, x, t, y_pred_ite, ts, num_units,
                                                           dropout, l2_weight, learning_rate,
                                                           num_layers, num_treatments, with_bn,
                                                           nonlinearity, initializer, reuse=False)

        d_ite_true = GANITEBuilder.build_ite_discriminator(input_dim, x, t, y_full, ts, num_units,
                                                           dropout, l2_weight, learning_rate,
                                                           num_layers, num_treatments, with_bn,
                                                           nonlinearity, initializer, reuse=True)

        return y_pred_ite, d_ite_pred, d_ite_true, z_i

    @staticmethod
    def build_ite_generator(input_dim, x, t, y_f, ts, num_units=128, dropout=0.0, l2_weight=0.0,
                            learning_rate=0.0001, num_layers=2,
                            num_treatments=2, with_bn=False, nonlinearity="elu",
                            initializer=tf.variance_scaling_initializer()):
        nonlinearity = get_nonlinearity_by_name(nonlinearity)
        with tf.variable_scope("g_ite",
                               initializer=initializer):
            z_i = tf.placeholder("float", shape=[None, num_treatments], name='z_i')
            mlp_input = tf.concat([x, z_i], axis=-1)

            if ts is not None:
                mlp_input = tf.concat([mlp_input, ts], axis=-1)

            x = build_mlp(mlp_input, num_layers, num_units, dropout, nonlinearity)[0]
            y_pred = tf.layers.dense(x, units=num_treatments, use_bias=True,
                                     bias_initializer=initializer)
            return y_pred, z_i

    @staticmethod
    def build_ite_discriminator(input_dim, x, t, y_pred, ts, num_units=128, dropout=0.0, l2_weight=0.0,
                                learning_rate=0.0001, num_layers=2,
                                num_treatments=2, with_bn=False, nonlinearity="elu",
                                initializer=tf.variance_scaling_initializer(),
                                reuse=False):
        nonlinearity = get_nonlinearity_by_name(nonlinearity)
        with tf.variable_scope("d_ite",
                               reuse=reuse,
                               initializer=initializer):
            mlp_input = tf.concat([x, y_pred], axis=-1)
            x = build_mlp(mlp_input, num_layers, num_units, dropout, nonlinearity)[0]
            y = tf.layers.dense(x, units=1, use_bias=True,
                                bias_initializer=initializer, activation=tf.nn.sigmoid)
            return y
