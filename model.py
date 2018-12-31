# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import layers
from utils import load_word_matrix, load_char_matrix
import pickle
import math

from rnn_cell.cifg_cell import CoupledInputForgetGateLSTMCell
from rnn_cell.weight_drop_cell import CustomizedLSTMCell

ID_GO = 331273
ID_PAD = 0
NUM_VOCAB = 331274


class Model(object):
    def __init__(self, parameter):
        self.parameter = parameter
        self.regularizer = layers.l2_regularizer(self.parameter["lambda"])

    def build_model(self):
        self._build_placeholder()

        data = None
        # Load word vocab and char vocab if we are using pretrained embedding
        if self.parameter['use_word_pretrained'] or self.parameter['use_char_pretrained']:
            with open('necessary.pkl', 'rb') as f:
                data = pickle.load(f)

        self._build_word_and_char_embedding(data)

        # 각각의 임베딩 값을 가져온다
        self._embeddings = []
        self._embeddings.append(tf.nn.embedding_lookup(self._embedding_matrix[0], self.morph))
        self._embeddings.append(tf.nn.embedding_lookup(self._embedding_matrix[1], self.character))

        # 음절을 이용한 임베딩 값을 구한다.
        character_embedding = tf.reshape(self._embeddings[1],
                                         [-1, self.parameter["word_length"], self.parameter["embedding"][1][2]])
        char_len = tf.reshape(self.character_len, [-1])

        # Dropout after embedding, before lstm layer
        if self.parameter["use_dropout_after_embedding"]:
            character_embedding = tf.nn.dropout(character_embedding, self.emb_dropout_keep_prob)

        character_emb_rnn = self._build_birnn_model(character_embedding, char_len, self.parameter["char_lstm_units"],
                                                    self.lstm_dropout_keep_prob, last=True, scope="char_layer")

        # 위에서 구한 모든 임베딩 값을 concat 한다.
        all_data_emb = self.ne_dict
        for i in range(0, len(self._embeddings) - 1):
            all_data_emb = tf.concat([all_data_emb, self._embeddings[i]], axis=2)
        all_data_emb = tf.concat([all_data_emb, character_emb_rnn], axis=2)

        if self.parameter["use_highway"]:
            all_data_emb = self._build_highway(all_data_emb,
                                               self.parameter["num_layers"],
                                               scope="highway")
        # Dropout after embedding, before lstm layer
        if self.parameter["use_dropout_after_embedding"]:
            all_data_emb = tf.nn.dropout(all_data_emb, self.emb_dropout_keep_prob)

        # 모든 데이터를 가져와서 Bi-RNN 실시
        sentence_output = self._build_birnn_model(all_data_emb, self.sequence, self.parameter["lstm_units"],
                                                  self.lstm_dropout_keep_prob, scope="all_data_layer")
        if self.parameter["use_self_attention"]:
            aligned_output = self._attention(sentence_output,
                                             self.parameter["lstm_units"],
                                             self.parameter["num_heads"],
                                             self.sequence,
                                             scope="attention")
            outputs = tf.concat([sentence_output, aligned_output], axis=2)
        else:
            outputs = sentence_output

        outputs = tf.nn.dropout(outputs, self.dropout_rate)
        # [b, t, 3*d] -> [b, t, C]
        logits = self._build_dense_layer(outputs)

        # crf layer
        crf_cost = self._build_crf_layer(logits)
        if self.parameter["use_reg_loss"]:
            reg_vars = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            reg_term = layers.apply_regularization(self.regularizer,
                                                   reg_vars)
            self.cost = crf_cost + reg_term
        else:
            self.cost = crf_cost

        self.train_op = self._build_output_layer(self.cost)

        # Exponential moving average
        if self.parameter["use_ema"]:
            var_ema = tf.train.ExponentialMovingAverage(decay=self.parameter["ema_decay_rate"])
            ema_op = var_ema.apply(tf.trainable_variables())
            with tf.control_dependencies([ema_op]):
                self.cost = tf.identity(self.cost)

    def _build_placeholder(self):
        self.morph = tf.placeholder(tf.int32, [None, None])
        self.ne_dict = tf.placeholder(tf.float32, [None, None, int(self.parameter["n_class"] / 2)])
        self.character = tf.placeholder(tf.int32, [None, None, None])
        self.dropout_rate = tf.placeholder(tf.float32)
        self.weight_dropout_keep_prob = tf.placeholder(tf.float32, shape=[],
                                                       name="weight_dropout_keep_prob")
        self.lstm_dropout_keep_prob = tf.placeholder(tf.float32, shape=[], name="lstm_dropout")
        self.sequence = tf.placeholder(tf.int32, [None])
        self.character_len = tf.placeholder(tf.int32, [None, None])
        self.label = tf.placeholder(tf.int32, [None, None])
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.emb_dropout_keep_prob = tf.placeholder(tf.float32, name='emb_dropout_keep_prob')
        self.dense_dropout_keep_prob = tf.placeholder(tf.float32, name='dense_dropout_keep_prob')

        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')  # For lr decay

        if self.parameter["use_lm"]:
            dims = tf.shape(self.morph)
            self.batch_size = dims[0]
            self.nsteps = dims[1]
            self.nchars = self.parameter["word_length"]
            go_tokens = tf.ones(shape=[self.batch_size, 1], dtype=tf.int32) * ID_GO
            eos_tokens = tf.ones(shape=[self.batch_size, 1], dtype=tf.int32) * ID_PAD
            zero_chars = tf.ones(shape=[self.batch_size, 1, self.nchars], dtype=tf.int32) * ID_PAD
            zero_ne_dict = tf.zeros(shape=[self.batch_size, 1, int(self.parameter["n_class"] / 2)])
            self.encoder_inputs = tf.concat([go_tokens, self.morph], axis=-1)
            self.encoder_targets = tf.concat([self.morph, eos_tokens], axis=-1)
            self.encoder_input_chars = tf.concat([zero_chars, self.character], axis=1)
            self.encoder_length = self.sequence + 1
            zero_length = tf.zeros(shape=[self.batch_size, 1], dtype=tf.int32)
            self.encoder_char_len = tf.concat([zero_length, self.character_len], axis=1)
            self.lm_ne_dict = tf.concat([zero_ne_dict, self.ne_dict], axis=1)

    def _build_embedding(self, n_tokens, dimension, name="embedding"):
        # Random init은 무조건 trainable=True가 되도록
        embedding_weights = tf.get_variable(
            name, [n_tokens, dimension],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer(),
            regularizer=self.regularizer
        )
        return embedding_weights

    def _build_embedding_pretrained(self, embedding_matrix, trainable, name='embedding'):
        embedding_weights = tf.get_variable(
            name, embedding_matrix.shape,
            dtype=tf.float32,
            initializer=tf.constant_initializer(embedding_matrix),
            regularizer=self.regularizer,
            trainable=trainable
        )
        return embedding_weights

    def _build_word_and_char_embedding(self, data):
        self._embedding_matrix = []
        for item in self.parameter["embedding"]:
            if item[0] == 'word' and self.parameter['use_word_pretrained']:
                print("Using word pretrained_embedding...")
                word_matrix = load_word_matrix('word_emb_dim_300.pkl', data['word'],
                                               self.parameter["word_embedding_size"],
                                               self.parameter["word_embedding_trainble"])
                self._embedding_matrix.append(
                    self._build_embedding_pretrained(embedding_matrix=word_matrix,
                                                     trainable=self.parameter["word_embedding_trainble"],
                                                     name='embedding_word_pretrained'))
            elif item[0] == 'character' and self.parameter['use_char_pretrained']:
                print("Using char pretrained_embedding...")
                char_matrix = load_char_matrix('char_emb_dim_300.pkl', data['character'],
                                               self.parameter["char_embedding_size"],
                                               self.parameter["char_embedding_trainble"])
                self._embedding_matrix.append(
                    self._build_embedding_pretrained(embedding_matrix=char_matrix,
                                                     trainable=self.parameter["char_embedding_trainble"],
                                                     name='embedding_char_pretrained'))
            else:
                self._embedding_matrix.append(self._build_embedding(item[1], item[2], name="embedding_" + item[0]))

    def _build_single_cell(self, lstm_units, target, keep_prob):
        # 1. Use weight dropout
        if self.parameter["use_custom_lstm_cell"]:
            cell = CustomizedLSTMCell(lstm_units, self.weight_dropout_keep_prob,
                                      initializer=tf.contrib.layers.xavier_initializer())
        else:
            cell = CoupledInputForgetGateLSTMCell(
                lstm_units,
                use_peepholes=True,
                initializer=tf.contrib.layers.xavier_initializer(),
                state_is_tuple=True)

        # 2. Use variational dropout
        if self.parameter["use_variational_dropout"]:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                 #  input_keep_prob=keep_prob,
                                                 output_keep_prob=keep_prob,
                                                 variational_recurrent=True,
                                                 input_size=target.shape[-1],
                                                 dtype=tf.float32)
        else:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                 #  input_keep_prob=keep_prob,
                                                 output_keep_prob=keep_prob)

        return cell

    def _build_birnn_model(self, target, seq_len, lstm_units, keep_prob, last=False, scope="layer", lm=False):
        with tf.variable_scope("forward_" + scope, reuse=tf.AUTO_REUSE):
            lstm_fw_cell = self._build_single_cell(lstm_units, target, keep_prob)

        with tf.variable_scope("backward_" + scope, reuse=tf.AUTO_REUSE):
            lstm_bw_cell = self._build_single_cell(lstm_units, target, keep_prob)

        with tf.variable_scope("birnn-lstm_" + scope, reuse=tf.AUTO_REUSE):
            _output = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, dtype=tf.float32,
                                                      inputs=target, sequence_length=seq_len, scope="rnn_" + scope)
            if last and not lm:
                _, ((_, output_fw), (_, output_bw)) = _output
                outputs = tf.concat([output_fw, output_bw], axis=1)
                outputs = tf.reshape(
                    outputs, shape=[-1, self.parameter["sentence_length"], 2 * lstm_units])
            elif last and lm:
                _, ((_, output_fw), (_, output_bw)) = _output
                outputs = tf.concat([output_fw, output_bw], axis=1)
                outputs = tf.reshape(
                    outputs, shape=[-1, self.parameter["sentence_length"] + 1, 2 * lstm_units])
            else:
                (output_fw, output_bw), _ = _output
                outputs = tf.concat([output_fw, output_bw], axis=2)

        return outputs

    def _attention(self, inputs, units, num_heads, sequence_length, scope):
        # multi-head dot product attention
        with tf.variable_scope(scope):
            Q_ = tf.layers.dense(inputs, units, use_bias=False,
                                 activation=tf.nn.relu,
                                 kernel_initializer=layers.variance_scaling_initializer(),
                                 kernel_regularizer=self.regularizer)
            K_ = tf.layers.dense(inputs, units, use_bias=False,
                                 activation=tf.nn.relu,
                                 kernel_initializer=layers.variance_scaling_initializer(),
                                 kernel_regularizer=self.regularizer)
            V_ = tf.layers.dense(inputs, units, use_bias=False,
                                 activation=tf.nn.relu,
                                 kernel_initializer=layers.variance_scaling_initializer(),
                                 kernel_regularizer=self.regularizer)
            Q = tf.concat(tf.split(Q_, num_heads, axis=2), axis=0)
            K = tf.concat(tf.split(K_, num_heads, axis=2), axis=0)
            V = tf.concat(tf.split(V_, num_heads, axis=2), axis=0)

            weights = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
            weights /= (units // num_heads) ** 0.5
            max_len = self.parameter["sentence_length"]
            key_masks = tf.sequence_mask(sequence_length, maxlen=max_len, dtype=tf.float32)  # [b,t]
            key_masks = tf.tile(key_masks, [num_heads, 1])  # [b*h, t]
            key_masks = tf.expand_dims(key_masks, axis=1)  # [b*h, 1, t]
            key_masks = tf.tile(key_masks, [1, max_len, 1])  # [b*h, t, t]

            paddings = tf.ones_like(weights) * (-2 ** 32 + 1)
            weights = tf.where(tf.equal(key_masks, 0), paddings, weights)
            if self.parameter["penalize_self_align"]:
                ones = tf.ones_like(weights)
                diag = tf.matrix_band_part(ones, 0, 0) * (-2 ** 32 + 1)
                weights += diag
            weights = tf.nn.softmax(weights)

            query_masks = tf.sequence_mask(sequence_length, maxlen=max_len, dtype=tf.float32)
            query_masks = tf.tile(query_masks, [num_heads, 1])
            query_masks = tf.expand_dims(query_masks, axis=2)
            weights *= query_masks

            weights = tf.nn.dropout(weights, self.dropout_rate)
            outputs = tf.matmul(weights, V)

            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
            return outputs

    def _build_dense_layer(self, outputs):
        activation = None
        if self.parameter["use_non_linear_on_dense"]:
            # activation = tf.nn.relu
            activation = tf.nn.tanh
        if self.parameter["use_additional_dense"]:
            outputs = tf.layers.dense(outputs,
                                      units=self.parameter["dense_unit_size"],
                                      activation=activation,
                                      use_bias=True,
                                      kernel_regularizer=self.regularizer,
                                      kernel_initializer=layers.xavier_initializer())
            outputs = tf.nn.dropout(outputs, self.dense_dropout_keep_prob)
        logits = tf.layers.dense(outputs,
                                 units=self.parameter["n_class"],
                                 activation=None,
                                 use_bias=True,
                                 kernel_regularizer=self.regularizer,
                                 kernel_initializer=layers.xavier_initializer())
        return logits

    def _build_crf_layer(self, target):
        with tf.variable_scope("crf_layer"):
            self.matricized_unary_scores = target
            self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                self.matricized_unary_scores, self.label, self.sequence)
            cost = tf.reduce_mean(-self.log_likelihood)
            self.viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(target,
                                                                             self.transition_params,
                                                                             self.sequence)
        return cost

    def _build_output_layer(self, cost):
        with tf.variable_scope("output_layer"):
            opt = tf.train.AdamOptimizer(self.learning_rate)
            if self.parameter["use_grad_clip"]:
                # apply grad clip to avoid gradient explosion
                grads_vars = opt.compute_gradients(cost)
                capped_grads_vars = [[tf.clip_by_value(g, -self.parameter["clip"], self.parameter["clip"]), v]
                                     for g, v in grads_vars]
                train_op = opt.apply_gradients(capped_grads_vars, self.global_step)
            else:
                train_op = opt.minimize(cost, global_step=self.global_step)
        return train_op

    def _build_highway(self, inputs, num_layers, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            outputs = inputs
            dim = outputs.shape[-1]
            for i in range(num_layers):
                trans = tf.layers.dense(outputs, dim, activation=tf.nn.relu,
                                        kernel_initializer=layers.variance_scaling_initializer(),
                                        kernel_regularizer=self.regularizer,
                                        name="trans_{}".format(i))
                trans = tf.nn.dropout(trans, self.dropout_rate)
                gate = tf.layers.dense(outputs, dim, activation=tf.nn.sigmoid,
                                       kernel_initializer=layers.xavier_initializer(),
                                       kernel_regularizer=self.regularizer,
                                       name="gate_{}".format(i))
                outputs = gate * trans + (1.0 - gate) * outputs
            return outputs


class ConvModel(Model):
    def build_model(self):
        self._build_placeholder()

        data = None
        # Load word vocab and char vocab if we are using pretrained embedding
        if self.parameter['use_word_pretrained'] or self.parameter['use_char_pretrained']:
            with open('necessary.pkl', 'rb') as f:
                data = pickle.load(f)

        self._build_word_and_char_embedding(data)

        # 각각의 임베딩 값을 가져온다
        self._embeddings = []
        self._embeddings.append(tf.nn.embedding_lookup(self._embedding_matrix[0], self.morph))
        self._embeddings.append(tf.nn.embedding_lookup(self._embedding_matrix[1], self.character))

        # 음절을 이용한 임베딩 값을 구한다.
        character_embedding = tf.reshape(self._embeddings[1],
                                         [-1, self.parameter["word_length"], self.parameter["embedding"][1][2]])
        char_len = tf.reshape(self.character_len, [-1])

        # Dropout after embedding, before lstm layer
        if self.parameter["use_dropout_after_embedding"]:
            character_embedding = tf.nn.dropout(character_embedding, self.emb_dropout_keep_prob)

        character_emb_rnn = self._build_birnn_model(character_embedding, char_len, self.parameter["char_lstm_units"],
                                                    self.lstm_dropout_keep_prob, last=True, scope="char_layer")

        if self.parameter["use_lm"]:
            lm_word_embedding = tf.nn.embedding_lookup(self._embedding_matrix[0], self.encoder_inputs)
            lm_char_embedding = tf.nn.embedding_lookup(self._embedding_matrix[1], self.encoder_input_chars)
            lm_char_embedding = tf.reshape(lm_char_embedding, [-1, self.parameter["word_length"],
                                                               self.parameter["embedding"][1][2]])
            lm_char_len = tf.reshape(self.encoder_char_len, [-1])
            lm_char_rnn = self._build_birnn_model(lm_char_embedding, lm_char_len, self.parameter["char_lstm_units"],
                                                  self.lstm_dropout_keep_prob, last=True, scope="char_layer", lm=True)
            lm_all_emb = tf.concat([self.lm_ne_dict, lm_word_embedding, lm_char_rnn], axis=2)
            if self.parameter["use_highway"]:
                lm_all_emb = self._build_highway(lm_all_emb, self.parameter["num_layers"],
                                                 scope="highway")

        # 위에서 구한 모든 임베딩 값을 concat 한다.
        all_data_emb = self.ne_dict
        for i in range(0, len(self._embeddings) - 1):
            all_data_emb = tf.concat([all_data_emb, self._embeddings[i]], axis=2)
        all_data_emb = tf.concat([all_data_emb, character_emb_rnn], axis=2)

        if self.parameter["use_highway"]:
            all_data_emb = self._build_highway(all_data_emb,
                                               self.parameter["num_layers"],
                                               scope="highway")

        # Dropout after embedding, before lstm layer
        if self.parameter["use_dropout_after_embedding"]:
            all_data_emb = tf.nn.dropout(all_data_emb, self.emb_dropout_keep_prob)

        output_lst = []
        # --------------------------------------- Add CONV Layer -------------------------------------------#
        # 1d depthwise-separable convolution
        if self.parameter["use_lm"]:
            conv_output = self._build_conv(lm_all_emb,
                                           self.parameter["kernel_sizes"],
                                           self.parameter["num_filters"],
                                           self.encoder_length,
                                           auto_regressive=True)
            self.lm_loss = self.lm_loss(conv_output, self.encoder_targets,
                                        NUM_VOCAB, self.encoder_length, scope="lm_loss")
            # remove go token
            conv_output = conv_output[:, 1:, :]
        else:
            conv_output = self._build_conv(all_data_emb,
                                           self.parameter["kernel_sizes"],
                                           self.parameter["num_filters"],
                                           self.sequence,
                                           auto_regressive=False)
        output_lst.append(conv_output)
        # --------------------------------------------------------------------------------------------------#

        # 모든 데이터를 가져와서 Bi-RNN 실시
        lstm_output = self._build_birnn_model(all_data_emb, self.sequence, self.parameter["lstm_units"],
                                              self.lstm_dropout_keep_prob, scope="all_data_layer")
        output_lst.append(lstm_output)

        # self attention
        if self.parameter["use_self_attention"]:
            aligned_outputs = self._attention(lstm_output,
                                              self.parameter["lstm_units"],
                                              self.parameter["num_heads"],
                                              self.sequence,
                                              scope="attention_small")
            output_lst.append(aligned_outputs)

        if len(output_lst) != 1:
            outputs = tf.concat(output_lst, axis=2)
        else:
            outputs = output_lst[0]

        outputs = tf.nn.dropout(outputs, self.dropout_rate)
        # [b, t, 3*d] -> [b, t, C]
        logits = self._build_dense_layer(outputs)

        # crf layer
        crf_cost = self._build_crf_layer(logits)
        if self.parameter["use_reg_loss"]:
            reg_vars = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            reg_term = layers.apply_regularization(self.regularizer,
                                                   reg_vars)
            self.cost = crf_cost + reg_term
        else:
            self.cost = crf_cost

        if self.parameter["use_lm"]:
            self.cost += self.lm_loss * self.parameter["gamma"]

        self.train_op = self._build_output_layer(self.cost)

        # Exponential moving average
        if self.parameter["use_ema"]:
            var_ema = tf.train.ExponentialMovingAverage(decay=self.parameter["ema_decay_rate"])
            ema_op = var_ema.apply(tf.trainable_variables())
            with tf.control_dependencies([ema_op]):
                self.cost = tf.identity(self.cost)

    def _build_conv(self, inputs, kernel_sizes, num_filters, sequence_length, auto_regressive):
        conv_lst = []
        for kernel_size in kernel_sizes:
            conv = self._depthwise_separable_conv(inputs, kernel_size,
                                                  num_filters,
                                                  sequence_length,
                                                  scope="conv_{}".format(kernel_size),
                                                  auto=auto_regressive)
            conv_lst.append(conv)
        outputs = tf.concat(conv_lst, axis=-1)
        return outputs

    def _depthwise_separable_conv(self, inputs, kernel_size, num_filters, sequence_length, scope, auto, reuse=False):
        # inputs : [b, t, d] -> [b, t, 1, d]
        dims = inputs.shape.as_list()
        padding = "SAME"
        maxlen = self.parameter["sentence_length"]
        inputs = tf.expand_dims(inputs, axis=2)
        if auto:
            # pad the inputs for auto-regressive
            zeros = tf.constant([[0, 0], [kernel_size - 1, 0], [0, 0], [0, 0]])
            inputs = tf.pad(inputs, zeros)
            padding = "VALID"
            maxlen = self.parameter["sentence_length"] + 1
        with tf.variable_scope(scope, reuse=reuse):
            depthwise_filter = tf.get_variable(shape=[kernel_size, 1, dims[-1], 1],
                                               name="depth_filter",
                                               initializer=layers.xavier_initializer(),
                                               regularizer=self.regularizer)
            pointwise_filter = tf.get_variable(shape=[1, 1, dims[-1], num_filters],
                                               name="point_filter",
                                               initializer=layers.xavier_initializer(),
                                               regularizer=self.regularizer)
            outputs = tf.nn.separable_conv2d(inputs, depthwise_filter, pointwise_filter,
                                             padding=padding, strides=(1, 1, 1, 1))
            # reshape to original dim [b, t, 1, d] -> [b,t,d] and apply layer norm
            outputs = tf.squeeze(outputs, axis=2)
            mask = tf.sequence_mask(sequence_length,
                                    maxlen=maxlen,
                                    dtype=tf.float32)
            mask = tf.expand_dims(mask, axis=2)
            outputs *= mask
            outputs = layers.layer_norm(outputs, begin_norm_axis=-1, begin_params_axis=-1)
            outputs = tf.nn.relu(outputs)
            if self.parameter["use_positional_embedding"]:
                outputs += self._position_embeddings(outputs, sequence_length, maxlen)
            return outputs

    def lm_loss(self, inputs, targets, num_words, sequence_length, scope):
        with tf.variable_scope(scope):
            dim = inputs.shape[-1]
            inputs = tf.reshape(inputs, [-1, dim])
            labels = tf.reshape(targets, [-1, 1])
            softmax_w = tf.get_variable(shape=[num_words, dim], name="w",
                                        initializer=layers.xavier_initializer(),
                                        regularizer=self.regularizer)
            softmax_b = tf.get_variable(shape=[num_words], name="b",
                                        initializer=tf.zeros_initializer())
            sampled_loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_b,
                                                      labels, inputs,
                                                      8192, num_words)
            maxlen = self.parameter["sentence_length"] + 1
            mask = tf.sequence_mask(sequence_length, maxlen=maxlen, dtype=tf.float32)
            mask = tf.reshape(mask, [-1])
            sampled_loss *= mask
            crossent = tf.reduce_sum(sampled_loss)
            crossent /= tf.reduce_sum(mask)

            return crossent

    def _position_embeddings(self, inputs, sequence_length, maxlen):
        length = tf.shape(inputs)[1]
        channels = tf.shape(inputs)[2]
        max_timescale = 1.0e4
        min_timescale = 1.0
        position = tf.cast(tf.range(length), tf.float32)
        num_timescales = channels // 2
        log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (tf.to_float(num_timescales) - 1))
        inv_timescales = min_timescale * tf.exp(tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
        signal = tf.reshape(signal, [1, length, channels])
        # mask for zero padding
        mask = tf.sequence_mask(sequence_length, maxlen=maxlen, dtype=tf.float32)
        mask = tf.expand_dims(mask, axis=2)
        signal *= mask
        return signal
