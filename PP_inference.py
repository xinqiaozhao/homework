import tensorflow as tf

HIDDEN_SIZE = 500
NUM_LAYERS = 2


def lstm_model(x,dropout_keep_prob):

    lstm_cells = [
        tf.nn.rnn_cell.DropoutWrapper(
            tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE),
            output_keep_prob=dropout_keep_prob)
        for _ in range(NUM_LAYERS)]
    cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)
    print("cell_created")
    outputs, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    output = outputs[:,-1,:]

    predictions01 = tf.contrib.layers.fully_connected(output, 100,activation_fn= tf.nn.relu)
    predictions = tf.contrib.layers.fully_connected(predictions01, 1, activation_fn=None)
    return predictions

