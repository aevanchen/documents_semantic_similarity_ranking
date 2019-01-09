import tensorflow as tf
import numpy as np

class SiameseLSTMw2v(object):
    """
    A LSTM based deep Siamese network for text similarity.
    Uses an word embedding layer (looks up in pre-trained w2v), followed by a biLSTM and Energy Loss layer.
    """
    #train model in main 
    #sess.run(siameseModel.W.assign(initW))
    
# word2vec_format="bin"
# embedding_dim=300
# dropout_keep_prob=1.0
# l2_reg_lambda=0.0
# hidden_units=50

    
#      siameseModel = SiameseLSTMw2v(
#                 sequence_length=max_document_length,
#                 vocab_size=len(vocab_processor.vocabulary_),
#                 embedding_size=FLAGS.embedding_dim,
#                 hidden_units=FLAGS.hidden_units,
#                 l2_reg_lambda=FLAGS.l2_reg_lambda,
#                 batch_size=FLAGS.batch_size,
#                 trainableEmbeddings=trainableEmbeddings
#             )
#         # Define Training procedure
#         global_step = tf.Variable(0, name="global_step", trainable=False)
#         optimizer = tf.train.AdamOptimizer(1e-3)
#         print("initialized siameseModel object")
    
  #  grads_and_vars=optimizer.compute_gradients(siameseModel.loss)
  #  tr_op_set = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    
    def stackedRNN(self, x, dropout, scope, embedding_size, sequence_length, hidden_units):
        n_hidden=hidden_units
        n_layers=3
        # Prepare data shape to match `static_rnn` function requirements
        x = tf.unstack(tf.transpose(x, perm=[1, 0, 2]))
        print(x[0].shape)
        # Define lstm cells with tensorflow
        # Forward direction cell

        with tf.name_scope("fw"+scope),tf.variable_scope("fw"+scope):
            stacked_rnn_fw = []
            for _ in range(n_layers):
                fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell,output_keep_prob=dropout)
                stacked_rnn_fw.append(lstm_fw_cell)
            lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)
            #print ("lstm shape is "+ lstm_fw_cell_m.shape)
            outputs, _ = tf.nn.static_rnn(lstm_fw_cell_m, x, dtype=tf.float32)
            #print ("output length is "+ len(outputs))
            print ("output shape is "+ str(outputs[-1].shape))
        return outputs[-1]

    def contrastive_loss(self, y,d,batch_size):
        tmp= y *tf.square(d) ## when they are almost the same y=1, the square means they are increasing 
        #tmp= tf.mul(y,tf.square(d))
        tmp2 = (1-y) *tf.square(tf.maximum((1 - d),0)) 
        ## hinge loss  when they are not same, the function is decreasing monotonically
        return tf.reduce_sum(tmp +tmp2)/batch_size/2
    
    def __init__(
        self, sequence_length, vocab_size, embedding_size, hidden_units, l2_reg_lambda, batch_size, trainableEmbeddings):

        # Placeholders for input, output and dropout
        self.input_x1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x1")
        self.input_x2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x2")
        self.input_y = tf.placeholder(tf.float32, [None], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0, name="l2_loss")
          
        # Embedding layer
        with tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.constant(0.0, shape=[vocab_size, embedding_size]),
                trainable=trainableEmbeddings,name="W")
            self.embedded_words1 = tf.nn.embedding_lookup(self.W, self.input_x1) # dim:(sequence,word_count,embed_dim)
            self.embedded_words2 = tf.nn.embedding_lookup(self.W, self.input_x2) # dim:(sequence,word_count,embed_dim)
        print ("embeded shape is "+str(self.embedded_words1.shape ))
        # Create a convolution + maxpool layer for each filter size
        with tf.name_scope("output"):
            self.out1=self.stackedRNN(self.embedded_words1, self.dropout_keep_prob, "side1", embedding_size, sequence_length, hidden_units)
            self.out2=self.stackedRNN(self.embedded_words2, self.dropout_keep_prob, "side2", embedding_size, sequence_length, hidden_units)
            self.distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.out1,self.out2)),1,keep_dims=True))
            self.distance = tf.div(self.distance, tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.out1),1,keep_dims=True)),tf.sqrt(tf.reduce_sum(tf.square(self.out2),1,keep_dims=True))))
            self.distance = tf.reshape(self.distance, [-1], name="distance")
        with tf.name_scope("loss"):
            self.loss = self.contrastive_loss(self.input_y,self.distance, batch_size)
        #### Accuracy computation is outside of this class.
        with tf.name_scope("accuracy"):
            self.temp_sim = tf.subtract(tf.ones_like(self.distance),tf.rint(self.distance), name="temp_sim") #auto threshold 0.5
            correct_predictions = tf.equal(self.temp_sim, self.input_y)
            self.accuracy=tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
