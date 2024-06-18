import tensorflow as tf
import config


class Critic(object):
    def __init__(self, n_node, node_emd_init):
        self.n_node = n_node
        self.node_emd_init = node_emd_init



        with tf.compat.v1.variable_scope('Critic'):
            self.embedding_matrix = tf.compat.v1.get_variable(name="embedding",
                                                    shape=self.node_emd_init.shape,
                                                    initializer=tf.constant_initializer(self.node_emd_init),
                                                    trainable=True)
            self.bias_vector = tf.Variable(tf.zeros([self.n_node]))

        self.node_id = tf.compat.v1.placeholder(tf.int32, shape = [None])
        self.node_neighbor_id = tf.compat.v1.placeholder(tf.int32, shape = [None])
        self.label = tf.compat.v1.placeholder(tf.float32, shape = [None])

        self.node_embedding = tf.nn.embedding_lookup(self.embedding_matrix, self.node_id)
        self.node_neighbor_embedding = tf.nn.embedding_lookup(self.embedding_matrix, self.node_neighbor_id)
        self.bias = tf.gather(self.bias_vector, self.node_neighbor_id)
        self.score = tf.reduce_sum(tf.multiply(self.node_embedding, self.node_neighbor_embedding), axis=1) + self.bias
        self.reward = tf.math.log(1 + tf.exp(self.score))


        with tf.GradientTape() as tape:
            self.loss = -tf.reduce_mean(tf.nn.sigmoid(self.score)) + tf.reduce_mean(tf.nn.sigmoid((self.score * self.reward)))
        optimizer = tf.compat.v1.train.RMSPropOptimizer(config.lr_critic)                                             
        self.c_updates = optimizer.minimize(self.loss)
        self.clip_weights_op = [w.assign(tf.clip_by_value(w, config.clipping[0], config.clipping[1])) for w in
                                [self.embedding_matrix, self.bias_vector]]
