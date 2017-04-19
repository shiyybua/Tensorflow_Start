import tflearn
import tensorflow as tf
from tensorflow.contrib import legacy_seq2seq
from tensorflow.contrib.rnn import core_rnn_cell
import numpy as np

class TFLearnSeq2Seq(object):
    def __init__(self, seq2seq_model=None, data_dir=None):
        '''
        seq2seq_model = string specifying which seq2seq model to use, e.g. "embedding_rnn"
        '''
        self.seq2seq_model = seq2seq_model or "embedding_rnn"
        self.in_seq_len = 10
        self.out_seq_len = 10
        self.in_max_int = 9
        self.out_max_int = 9
        self.n_input_symbols = self.in_max_int + 1
        self.n_output_symbols = self.out_max_int + 2		# extra one for GO symbol
        self.model_instance = None
        self.data_dir = data_dir

    def sequence_loss(self, y_pred, y_true):
        '''
        Loss function for the seq2seq RNN.  Reshape predicted and true (label) tensors, generate dummy weights,
        then use seq2seq.sequence_loss to actually compute the loss function.
        '''
        logits = tf.unstack(y_pred, axis=1)		# list of [-1, num_decoder_synbols] elements
        targets = tf.unstack(y_true, axis=1)		# y_true has shape [-1, self.out_seq_len]; unpack to list of self.out_seq_len [-1] elements
        weights = [tf.ones_like(yp, dtype=tf.float32) for yp in targets]
        sl = legacy_seq2seq.sequence_loss(logits, targets, weights)
        return sl


    def model(self, mode="train", num_layers=2, cell_size=32, cell_type="BasicLSTMCell", embedding_size=20,
              learning_rate=0.0001,
              tensorboard_verbose=0, checkpoint_path=None):

        assert mode in ["train", "predict"]

        checkpoint_path = checkpoint_path or (
        "%s%ss2s_checkpoint.tfl" % (self.data_dir or "", "/" if self.data_dir else ""))
        GO_VALUE = self.out_max_int + 1  # unique integer value used to trigger decoder outputs in the seq2seq RNN

        network = tflearn.input_data(shape=[None, self.in_seq_len + self.out_seq_len], dtype=tf.int32, name="XY")
        encoder_inputs = tf.slice(network, [0, 0], [-1, self.in_seq_len], name="enc_in")  # get encoder inputs
        encoder_inputs = tf.unstack(encoder_inputs, axis=1)  # transform into list of self.in_seq_len elements, each [-1]

        decoder_inputs = tf.slice(network, [0, self.in_seq_len], [-1, self.out_seq_len],
                                  name="dec_in")  # get decoder inputs
        decoder_inputs = tf.unstack(decoder_inputs, axis=1)  # transform into list of self.out_seq_len elements, each [-1]

        go_input = tf.multiply(tf.ones_like(decoder_inputs[0], dtype=tf.int32),
                               GO_VALUE)  # insert "GO" symbol as the first decoder input; drop the last decoder input
        decoder_inputs = [go_input] + decoder_inputs[: self.out_seq_len - 1]  # insert GO as first; drop last decoder input

        feed_previous = not (mode == "train")

        self.n_input_symbols = self.in_max_int + 1  # default is integers from 0 to 9
        self.n_output_symbols = self.out_max_int + 2  # extra "GO" symbol for decoder inputs

        single_cell = getattr(core_rnn_cell, cell_type)(cell_size, state_is_tuple=True)
        if num_layers == 1:
            cell = single_cell
        else:
            cell = core_rnn_cell.MultiRNNCell([single_cell] * num_layers)

        if self.seq2seq_model == "embedding_rnn":
            model_outputs, states = legacy_seq2seq.embedding_rnn_seq2seq(encoder_inputs,
                                                                         # encoder_inputs: A list of 2D Tensors [batch_size, input_size].
                                                                         decoder_inputs,
                                                                         cell,
                                                                         num_encoder_symbols=self.n_input_symbols,
                                                                         num_decoder_symbols=self.n_output_symbols,
                                                                         embedding_size=embedding_size,
                                                                         feed_previous=feed_previous)
        elif self.seq2seq_model == "embedding_attention":
            model_outputs, states = legacy_seq2seq.embedding_attention_seq2seq(encoder_inputs,
                                                                               # encoder_inputs: A list of 2D Tensors [batch_size, input_size].
                                                                               decoder_inputs,
                                                                               cell,
                                                                               num_encoder_symbols=self.n_input_symbols,
                                                                               num_decoder_symbols=self.n_output_symbols,
                                                                               embedding_size=embedding_size,
                                                                               num_heads=1,
                                                                               initial_state_attention=False,
                                                                               feed_previous=feed_previous)
        else:
            raise Exception('[TFLearnSeq2Seq] Unknown seq2seq model %s' % self.seq2seq_model)

        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + "seq2seq_model",
                             model_outputs)  # for TFLearn to know what to save and restore

        # model_outputs: list of the same length as decoder_inputs of 2D Tensors with shape [batch_size x output_size] containing the generated outputs.

        network = tf.stack(model_outputs, axis=1)  # shape [-1, n_decoder_inputs (= self.out_seq_len), num_decoder_symbols]

        with tf.name_scope("TargetsData"):  # placeholder for target variable (i.e. trainY input)
            targetY = tf.placeholder(shape=[None, self.out_seq_len], dtype=tf.int32, name="Y")

        network = tflearn.regression(network,
                                     placeholder=targetY,
                                     optimizer='adam',
                                     learning_rate=learning_rate,
                                     loss=self.sequence_loss,
                                     metric=self.accuracy,
                                     name="Y")

        model = tflearn.DNN(network, tensorboard_verbose=tensorboard_verbose, checkpoint_path=checkpoint_path)
        return model

    def train(self, num_epochs=20, num_points=100000,
              validation_set=0.1, snapshot_step=5000, batch_size=128):
        '''
        Train model, with specified number of epochs, and dataset size.

        Use specified model, or create one if not provided.  Load initial weights from file weights_input_fn,
        if provided. validation_set specifies what to use for the validation.

        Returns logits for prediction, as an numpy array of shape [out_seq_len, n_output_symbols].
        '''
        trainXY, trainY = self.generate_trainig_data(num_points)

        model = self.model()

        model.fit(trainXY, trainY,
                  n_epoch=num_epochs,
                  validation_set=validation_set,
                  batch_size=batch_size,
                  shuffle=True,
                  show_metric=False,
                  snapshot_step=snapshot_step,
                  snapshot_epoch=False,
                  run_id="TFLearnSeq2Seq"
                  )
        print ("Done!")

        return model

    def generate_trainig_data(self, num_points):
        '''
        Generate training dataset.  Produce random (integer) sequences X, and corresponding
        expected output sequences Y = generate_output_sequence(X).

        Return xy_data, y_data (both of type uint32)

        xy_data = numpy array of shape [num_points, in_seq_len + out_seq_len], with each point being X + Y
        y_data  = numpy array of shape [num_points, out_seq_len]
        '''
        x_data = np.random.randint(0, self.in_max_int, size=(num_points, self.in_seq_len))		# shape [num_points, in_seq_len]
        x_data = x_data.astype(np.uint32)						# ensure integer type

        y_data = [x for x in x_data ]
        y_data = np.array(y_data)

        xy_data = np.append(x_data, y_data, axis=1)		# shape [num_points, 2*seq_len]
        return xy_data, y_data

    def accuracy(self, y_pred, y_true, x_in):		# y_pred is [-1, self.out_seq_len, num_decoder_symbols]; y_true is [-1, self.out_seq_len]
        '''
        Compute accuracy of the prediction, based on the true labels.  Use the average number of equal
        values.
        '''
        pred_idx = tf.to_int32(tf.argmax(y_pred, 2))		# [-1, self.out_seq_len]
        accuracy = tf.reduce_mean(tf.cast(tf.equal(pred_idx, y_true), tf.float32), name='acc')
        return accuracy

    def predict(self, Xin, weights_input_fn=None):
        '''
        Make a prediction, using the seq2seq model, for the given input sequence Xin.
        If model is not provided, create one (or use last created instance).

        Return prediction, y

        prediction = array of integers, giving output prediction.  Length = out_seq_len
        y = array of shape [out_seq_len, out_max_int], giving logits for output prediction
        '''

        model = self.model('predict')
        model.load('./model/s2s_checkpoint.tfl-120',weights_only=True)

        X = np.array(Xin).astype(np.uint32)

        assert len(X)==self.in_seq_len
        Yin = [0]*self.out_seq_len

        XY = np.append(X, np.array(Yin).astype(np.float32))
        XY = XY.reshape([-1, self.in_seq_len + self.out_seq_len])		# batch size 1

        res = model.predict(XY)
        res = np.array(res)
        y = res.reshape(self.out_seq_len, self.n_output_symbols)
        prediction = np.argmax(y, axis=1)
        print prediction
        return prediction, y


seq = TFLearnSeq2Seq("embedding_attention")
seq.train(num_epochs=20, num_points=1000,
              validation_set=0.1, snapshot_step=20, batch_size=128)

# seq.predict([1,2,4,5,6,7,1,2,4,5])