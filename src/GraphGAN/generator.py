import tensorflow as tf
import config


class Generator(object):
    def __init__(self, n_node, node_emd_init):
        self.n_node = n_node
        self.node_emd_init = node_emd_init

        with tf.compat.v1.variable_scope('generator'):
            self.embedding_matrix = tf.compat.v1.get_variable(name="embedding",
                                                    shape=self.node_emd_init.shape,
                                                    initializer=tf.constant_initializer(self.node_emd_init),
                                                    trainable=True,
                                                    dtype=tf.float32)
            self.bias_vector = tf.Variable(tf.zeros([self.n_node]))

        self.node_id = tf.compat.v1.placeholder(tf.int32, shape = [None])
        self.node_neighbor_id = tf.compat.v1.placeholder(tf.int32, shape = [None] )
        self.reward = tf.compat.v1.placeholder(tf.float32, shape = [None])

        self.all_score = tf.matmul(self.embedding_matrix, self.embedding_matrix, transpose_b=True) + self.bias_vector
        self.node_embedding = tf.nn.embedding_lookup(self.embedding_matrix, self.node_id)  # batch_size * n_embed
        self.node_neighbor_embedding = tf.nn.embedding_lookup(self.embedding_matrix, self.node_neighbor_id)
        self.bias = tf.gather(self.bias_vector, self.node_neighbor_id)
        self.score = tf.reduce_sum(self.node_embedding * self.node_neighbor_embedding, axis=1) + self.bias

        with tf.GradientTape() as tape:
          self.loss = -tf.reduce_mean(tf.nn.sigmoid(self.score * self.reward))
        optimizer = tf.compat.v1.train.RMSPropOptimizer(config.lr_gen)
        self.g_updates = optimizer.minimize(self.loss)
        self.clip_weights_op = [w.assign(tf.clip_by_value(w, config.clipping[0], config.clipping[1])) for w in
                                [self.embedding_matrix, self.bias_vector]]
