import tensorflow as tf
print(tf.__version__)


import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

DEVICE = "cuda" if tf.test.is_gpu_available() else "cpu"
print("Running on device:", DEVICE.upper())


#copy and paste configuration of hyperparameters
modes = ["gen", "dis"]

# training settings
batch_size_gen = 64  # batch size for the generator
batch_size_dis = 64  # batch size for the discriminator
lambda_gen = 1e-5  # l2 loss regulation weight for the generator
lambda_dis = 1e-5  # l2 loss regulation weight for the discriminator
n_sample_gen = 20  # number of samples for the generator
lr_gen = 5e-5  # learning rate for the generator
lr_critic = 5e-5  # learning rate for the discriminator
n_epochs = 40 # number of outer loops
clipping = [-0.01, 0.01] # clipping parameters to ensure Lipschitz continuity
n_epochs_gen = 10  # number of inner loops for the generator
n_epochs_critic = 5  # number of inner loops of the critic
gen_interval = n_epochs_gen  # sample new nodes for the generator for every gen_interval iterations
dis_interval = n_epochs_critic  # sample new nodes for the discriminator for every dis_interval iterations
update_ratio = 1    # updating ratio when choose the trees

# model saving
load_model = False  # whether loading existing model for initialization
save_steps = 10

# other hyper-parameters

n_embed = 50
multi_processing = False  # whether using multi-processing to construct BFS-trees
window_size = 2

app = "link_prediction"

# path settings

train_filename = r"C:\Users\vansh\AI and ML reading material\GraphGAN_Project\GraphGAN\CA-GrQc Dataset\CA-GrQc_train.txt"
test_filename = r"C:\Users\vansh\AI and ML reading material\GraphGAN_Project\GraphGAN\CA-GrQc Dataset\CA-GrQc_test.txt"
test_neg_filename = r"C:\Users\vansh\AI and ML reading material\GraphGAN_Project\GraphGAN\CA-GrQc Dataset\CA-GrQc_test_neg.txt"
pretrain_emb_filename_d = r"C:\Users\vansh\AI and ML reading material\GraphGAN_Project\GraphGAN\test_embeddings.emb"
pretrain_emb_filename_g = r"C:\Users\vansh\AI and ML reading material\GraphGAN_Project\GraphGAN\test_embeddings.emb"
emb_filenames = [r"C:\Users\vansh\AI and ML reading material\GraphGAN_Project\GraphGAN\Pre-Train Embeddings\gen.emb",
                 r"C:\Users\vansh\AI and ML reading material\GraphGAN_Project\GraphGAN\Pre-Train Embeddings\dis.emb"]
result_filename = "/scratch/vanshg.phy21.iitbhu/GraphGAN/bio_grid_human_results.txt"
cache_filename = r"C:\Users\vansh\AI and ML reading material\GraphGAN_Project\GraphGAN\cache_caqrgc.pkl"
model_log = "/scratch/vanshg.phy21.iitbhu/GraphGAN/log/FF"
cm_file = "/scratch/vanshg.phy21.iitbhu/GraphGAN/cm_file.txt"

import numpy as np


def str_list_to_float(str_list):
    return [float(item) for item in str_list]


def str_list_to_int(str_list):
    return [int(item) for item in str_list]


def read_edges(train_filename, test_filename):
    """read data from files

    Args:
        train_filename: training file name
        test_filename: test file name

    Returns:
        node_num: int, number of nodes in the graph
        graph: dict, node_id -> list of neighbors in the graph
    """

    graph = {}
    nodes = set()
    train_edges = read_edges_from_file(train_filename)
    test_edges = read_edges_from_file(test_filename) if test_filename != "" else []

    for edge in train_edges:
        nodes.add(edge[0])
        nodes.add(edge[1])
        if graph.get(edge[0]) is None:
            graph[edge[0]] = []
        if graph.get(edge[1]) is None:
            graph[edge[1]] = []
        graph[edge[0]].append(edge[1])
        graph[edge[1]].append(edge[0])

    for edge in test_edges:
        nodes.add(edge[0])
        nodes.add(edge[1])
        if graph.get(edge[0]) is None:
            graph[edge[0]] = []
        if graph.get(edge[1]) is None:
            graph[edge[1]] = []

    return len(nodes), graph


def read_edges_from_file(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        edges = [str_list_to_int(line.split()) for line in lines]
    return edges


def read_embeddings(filename, n_node, n_embed):
    """read pretrained node embeddings
    """

    with open(filename, "r") as f:
        lines = f.readlines()[1:]  # skip the first line

        rng = np.random.default_rng(seed=0)

        embedding_matrix = rng.standard_normal(size = (n_node, n_embed))
        for line in lines:
            emd = line.split()
            embedding_matrix[int(emd[0]), :] = str_list_to_float(emd[1:])
        return embedding_matrix


def reindex_node_id(edges):
    """reindex the original node ID to [0, node_num)

    Args:
        edges: list, element is also a list like [node_id_1, node_id_2]
    Returns:
        new_edges: list[[1,2],[2,3]]
        new_nodes: list [1,2,3]
    """

    node_set = set()
    for edge in edges:
        node_set = node_set.union(set(edge))

    node_set = list(node_set)
    new_nodes = set()
    new_edges = []
    for edge in edges:
        new_edges.append([node_set.index(edge[0]), node_set.index(edge[1])])
        new_nodes = new_nodes.add(node_set.index(edge[0]))
        new_nodes = new_nodes.add(node_set.index(edge[1]))

    new_nodes = list(new_nodes)
    return new_edges, new_nodes


def generate_neg_links(train_filename, test_filename, test_neg_filename):
    """
    generate neg links for link prediction evaluation
    Args:
        train_filename: the training edges
        test_filename: the test edges
        test_neg_filename: the negative edges for test
    """

    train_edges = read_edges_from_file(train_filename)
    test_edges = read_edges_from_file(test_filename)
    neighbors = {}  # dict, node_ID -> list_of_neighbors
    for edge in train_edges + test_edges:
        if neighbors.get(edge[0]) is None:
            neighbors[edge[0]] = []
        if neighbors.get(edge[1]) is None:
            neighbors[edge[1]] = []
        neighbors[edge[0]].append(edge[1])
        neighbors[edge[1]].append(edge[0])
    nodes = set([x for x in range(len(neighbors))])

    # for each edge in the test set, sample a negative edge
    neg_edges = []

    for i in range(len(test_edges)):
        edge = test_edges[i]
        start_node = edge[0]
        neg_nodes = list(nodes.difference(set(neighbors[edge[0]] + [edge[0]])))
        neg_node = np.random.choice(neg_nodes, size=1)[0]
        neg_edges.append([start_node, neg_node])
    neg_edges_str = [str(x[0]) + "\t" + str(x[1]) + "\n" for x in neg_edges]
    with open(test_neg_filename, "w+") as f:
        f.writelines(neg_edges_str)

def load_pkl(filename):
    with open(filename+'.pkl', 'rb') as fr:
        return pickle.load(fr)

def softmax(x):
    e_x = np.exp(x - np.max(x))  # for computation stability
    return e_x / e_x.sum()

#discriminator network
import tensorflow as tf


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
            self.loss = -tf.reduce_mean(tf.nn.sigmoid(self.score)) + tf.reduce_mean(tf.nn.sigmoid((self.score * self.reward))) + lambda_dis * (
                tf.nn.l2_loss(self.node_neighbor_embedding) +
                tf.nn.l2_loss(self.node_embedding) +
                tf.nn.l2_loss(self.bias))
        optimizer = tf.compat.v1.train.RMSPropOptimizer(lr_critic)                                             
        self.c_updates = optimizer.minimize(self.loss)
        self.clip_weights_op = [w.assign(tf.clip_by_value(w, clipping[0], clipping[1])) for w in
                                [self.embedding_matrix, self.bias_vector]]


#generator network
import tensorflow as tf
#import config


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
          self.loss = -tf.reduce_mean(tf.nn.sigmoid(self.score * self.reward)) + lambda_gen * (
                tf.nn.l2_loss(self.node_neighbor_embedding) + tf.nn.l2_loss(self.node_embedding))
        optimizer = tf.compat.v1.train.RMSPropOptimizer(lr_gen)
        self.g_updates = optimizer.minimize(self.loss)
        self.clip_weights_op = [w.assign(tf.clip_by_value(w, clipping[0], clipping[1])) for w in
                                [self.embedding_matrix, self.bias_vector]]
        


"""
The class is used to evaluate the application of link prediction
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



class LinkPredictEval(object):
    def __init__(self, embed_filename, test_filename, test_neg_filename, n_node, n_embed):
        self.embed_filename = embed_filename  # each line: node_id, embeddings(dim: n_embed)
        self.test_filename = test_filename  # each line: node_id1, node_id2
        self.test_neg_filename = test_neg_filename  # each line: node_id1, node_id2
        self.n_node = n_node
        self.n_embed = n_embed
        self.emd = read_embeddings(embed_filename, n_node=n_node, n_embed=n_embed)

    def eval_link_prediction(self):
        test_edges = read_edges_from_file(self.test_filename)
        test_edges_neg = read_edges_from_file(self.test_neg_filename)
        test_edges.extend(test_edges_neg)

        # may exists isolated point
        score_res = []
        for i in range(len(test_edges)):
            score_res.append(np.dot(self.emd[test_edges[i][0]], self.emd[test_edges[i][1]]))
        print(type(score_res))
        test_label = np.array(score_res)
        median = np.mean(test_label)
        index_pos = test_label >= median
        index_neg = test_label < median
        test_label[index_pos] = 1
        test_label[index_neg] = 0
        
        true_label = np.zeros(test_label.shape)
        true_label[0: len(true_label) // 2] = 1

        print("test_label:", test_label)
        print("true_label:", true_label)
        accuracy = accuracy_score(true_label, test_label)
        precision = precision_score(true_label, test_label)
        recall = recall_score(true_label, test_label)
        f1score = f1_score(true_label, test_label)

        return accuracy, precision, recall, f1score, true_label, test_label



tf.compat.v1.reset_default_graph()
tf.compat.v1.disable_eager_execution()

#graphGAN framework
import os
import collections
import tqdm
import multiprocessing
import pickle
import numpy as np
import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.reset_default_graph()


class GraphGAN(object):
    def __init__(self):
        print("reading graphs...")
        self.n_node, self.graph = read_edges(train_filename, test_filename)
        self.root_nodes = [i for i in range(self.n_node)]

        print("reading initial embeddings...")
        self.node_embed_init_d = read_embeddings(filename=pretrain_emb_filename_d,
                                                       n_node=self.n_node,
                                                       n_embed=n_embed)
        self.node_embed_init_g = read_embeddings(filename=pretrain_emb_filename_g,
                                                       n_node=self.n_node,
                                                       n_embed=n_embed)
	

        # construct or read BFS-trees
        self.trees = None
        if os.path.isfile(cache_filename):
            print("reading BFS-trees from cache...")
            pickle_file = open(cache_filename, 'rb')
            self.trees = pickle.load(pickle_file)
            pickle_file.close()
        else:
            print("constructing BFS-trees...")
            pickle_file = open(cache_filename, 'wb')
            if multi_processing:
                self.construct_trees_with_mp(self.root_nodes)
            else:
                self.trees = self.construct_trees(self.root_nodes)
            pickle.dump(self.trees, pickle_file)
            pickle_file.close()

        print("building GAN model...")
        self.discriminator = None
        self.generator = None
        self.build_generator()
        self.build_Critic()

        self.latest_checkpoint = tf.train.latest_checkpoint(model_log)
        self.saver = tf.compat.v1.train.Saver()

        self.config = tf.compat.v1.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.init_op = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
        self.sess = tf.compat.v1.Session(config=self.config)
        self.sess.run(self.init_op)

    def construct_trees_with_mp(self, nodes):
        """use the multiprocessing to speed up trees construction

        Args:
            nodes: the list of nodes in the graph
        """

        cores = multiprocessing.cpu_count() // 2
        pool = multiprocessing.Pool(cores)
        new_nodes = []
        n_node_per_core = self.n_node // cores
        for i in range(cores):
            if i != cores - 1:
                new_nodes.append(nodes[i * n_node_per_core: (i + 1) * n_node_per_core])
            else:
                new_nodes.append(nodes[i * n_node_per_core:])
        self.trees = {}
        trees_result = pool.map(self.construct_trees, new_nodes)
        for tree in trees_result:
            self.trees.update(tree)

    def construct_trees(self, nodes):
        """use BFS algorithm to construct the BFS-trees

        Args:
            nodes: the list of nodes in the graph
        Returns:
            trees: dict, root_node_id -> tree, where tree is a dict: node_id -> list: [father, child_0, child_1, ...]
        """

        trees = {}
        for root in tqdm.tqdm(nodes):
            trees[root] = {}
            trees[root][root] = [root]
            used_nodes = set()
            queue = collections.deque([root])
            while len(queue) > 0:
                cur_node = queue.popleft()
                used_nodes.add(cur_node)
                for sub_node in self.graph[cur_node]:
                    if sub_node not in used_nodes:
                        trees[root][cur_node].append(sub_node)
                        trees[root][sub_node] = [cur_node]
                        queue.append(sub_node)
                        used_nodes.add(sub_node)
        return trees

    def build_generator(self):
        """initializing the generator"""

        with tf.compat.v1.variable_scope("generator"):
            self.generator = Generator(n_node=self.n_node, node_emd_init=self.node_embed_init_g)

    def build_Critic(self):
        """initializing the Critic"""

        with tf.compat.v1.variable_scope("Critic"):
            self.discriminator = Critic(n_node=self.n_node, node_emd_init=self.node_embed_init_d)

    def train(self):
        # restore the model from the latest checkpoint if exists
        checkpoint = tf.train.get_checkpoint_state(model_log)
        if checkpoint and checkpoint.model_checkpoint_path and load_model:
            print("loading the checkpoint: %s" % checkpoint.model_checkpoint_path)
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)

        self.write_embeddings_to_file()

        self.evaluation(self)


        print("start training...")
        for epoch in range(n_epochs):
            print("epoch %d" % epoch)

            # save the model
            if epoch > 0 and epoch % save_steps == 0:
                self.saver.save(self.sess, model_log + "model.checkpoint")

            for g_epoch in range(n_epochs_gen):
            
                # D-steps
                center_nodes = np.empty
                neighbor_nodes = []
                labels = []
                for d_epoch in range(n_epochs_critic):
                    # generate new nodes for the discriminator for every dis_interval iterations
                    if d_epoch % dis_interval == 0:
                        center_nodes, neighbor_nodes, labels = self.prepare_data_for_c()
                    # training
                    train_size = len(center_nodes)
                    start_list = list(range(0, train_size, batch_size_dis))
                    np.random.shuffle(start_list)
                    for start in start_list:
                        end = start + batch_size_dis
                        self.sess.run(fetches = self.discriminator.c_updates,
                                    feed_dict={self.discriminator.node_id: np.array(center_nodes[start:end]),
                                                self.discriminator.node_neighbor_id: np.array(neighbor_nodes[start:end]),
                                                self.discriminator.label: np.array(labels[start:end])
                                                })
                        self.sess.run(self.discriminator.clip_weights_op)

                    
                
                # G-steps
                node_1 = []
                node_2 = []
                reward = []
                if g_epoch % gen_interval == 0:
                    node_1, node_2, reward = self.prepare_data_for_g()

                # training
                train_size = len(node_1)
                start_list = list(range(0, train_size, batch_size_gen))
                np.random.shuffle(start_list)
                for start in start_list:
                    end = start + batch_size_gen
                    self.sess.run(fetches = self.generator.g_updates,
                                feed_dict={self.generator.node_id: np.array(node_1[start:end]),
                                            self.generator.node_neighbor_id: np.array(node_2[start:end]),
                                            self.generator.reward: np.array(reward[start:end])})
                    self.sess.run(self.generator.clip_weights_op)

            self.write_embeddings_to_file()
            self.evaluation(self) #test data results
        print("training completes")
    print("1")
    def prepare_data_for_c(self):
        """generate positive and negative samples for the discriminator, and record them in the txt file"""

        center_nodes = []
        neighbor_nodes = []
        labels = []
        for i in self.root_nodes:
            if np.random.rand() < update_ratio:
                pos = self.graph[i]
                neg, _ = self.sample(i, self.trees[i], len(pos), for_d=True)
                if len(pos) != 0 and neg is not None:
                    # positive samples
                    center_nodes.extend([i] * len(pos))
                    neighbor_nodes.extend(pos)
                    labels.extend([1] * len(pos))

                    # negative samples
                    center_nodes.extend([i] * len(pos))
                    neighbor_nodes.extend(neg)
                    labels.extend([0] * len(neg))
        return center_nodes, neighbor_nodes, labels
    print("2")
    def prepare_data_for_g(self):
        """sample nodes for the generator"""

        paths = []
        for i in self.root_nodes:
            if np.random.rand() < update_ratio:
                sample, paths_from_i = self.sample(i, self.trees[i], n_sample_gen, for_d=False)
                if paths_from_i is not None:
                    paths.extend(paths_from_i)
        node_pairs = list(map(self.get_node_pairs_from_path, paths))
        node_1 = []
        node_2 = []
        for i in range(len(node_pairs)):
            for pair in node_pairs[i]:
                node_1.append(pair[0])
                node_2.append(pair[1])
        reward = self.sess.run(self.discriminator.reward,
                               feed_dict={self.discriminator.node_id: np.array(node_1),
                                          self.discriminator.node_neighbor_id: np.array(node_2)})
        return node_1, node_2, reward
    print("3")
    def sample(self, root, tree, sample_num, for_d):
        """ sample nodes from BFS-tree

        Args:
            root: int, root node
            tree: dict, BFS-tree
            sample_num: the number of required samples
            for_d: bool, whether the samples are used for the generator or the discriminator
        Returns:
            samples: list, the indices of the sampled nodes
            paths: list, paths from the root to the sampled nodes
        """

        all_score = self.sess.run(self.generator.all_score)
        samples = []
        paths = []
        n = 0

        while len(samples) < sample_num:
            current_node = root
            previous_node = -1
            paths.append([])
            is_root = True
            paths[n].append(current_node)
            while True:
                node_neighbor = tree[current_node][1:] if is_root else tree[current_node]
                is_root = False
                if len(node_neighbor) == 0:  # the tree only has a root
                    return None, None
                if for_d:  # skip 1-hop nodes (positive samples)
                    if node_neighbor == [root]:
                        # in current version, None is returned for simplicity
                        return None, None
                    if root in node_neighbor:
                        node_neighbor.remove(root)
                relevance_probability = all_score[current_node, node_neighbor]
                relevance_probability = softmax(relevance_probability)
                next_node = np.random.choice(node_neighbor, size=1, p=relevance_probability)[0]  # select next node
                paths[n].append(next_node)
                if next_node == previous_node:  # terminating condition
                    samples.append(current_node)
                    break
                previous_node = current_node
                current_node = next_node
            n = n + 1
        return samples, paths
    print("4")
    @staticmethod
    def get_node_pairs_from_path(path):
        """
        given a path from root to a sampled node, generate all the node pairs within the given windows size
        e.g., path = [1, 0, 2, 4, 2], window_size = 2 -->
        node pairs= [[1, 0], [1, 2], [0, 1], [0, 2], [0, 4], [2, 1], [2, 0], [2, 4], [4, 0], [4, 2]]
        :param path: a path from root to the sampled node
        :return pairs: a list of node pairs
        """

        path = path[:-1]
        pairs = []
        for i in range(len(path)):
            center_node = path[i]
            for j in range(max(i - window_size, 0), min(i + window_size + 1, len(path))):
                if i == j:
                    continue
                node = path[j]
                pairs.append([center_node, node])
        return pairs
    print("5")
    def write_embeddings_to_file(self):
        """write embeddings of the generator and the discriminator to files"""

        modes = [self.generator, self.discriminator]
        for i in range(2):
            embedding_matrix = self.sess.run(modes[i].embedding_matrix)
            index = np.array(range(self.n_node)).reshape(-1, 1)
            embedding_matrix = np.hstack([index, embedding_matrix])
            embedding_list = embedding_matrix.tolist()
            embedding_str = [str(int(emb[0])) + "\t" + "\t".join([str(x) for x in emb[1:]]) + "\n"
                             for emb in embedding_list]
            with open(emb_filenames[i], "w+") as f:
                lines = [str(self.n_node) + "\t" + str(n_embed) + "\n"] + embedding_str
                f.writelines(lines)
    print("6")
    @staticmethod
    def evaluation(self):
        results = []
        if app == "link_prediction":
            for i in range(2):
                lpe = LinkPredictEval(
                    emb_filenames[i], test_filename, test_neg_filename, self.n_node, n_embed)
                result ,precision, recall, f1score, _, _ = lpe.eval_link_prediction()
                results.append(modes[i] + "_accuracy:" + str(result) + "\n")
                results.append(modes[i] + "_precision" + ":" + str(precision) + "\n")
                results.append(modes[i] + "_recall" + ":" + str(recall) + "\n")
                results.append(modes[i] + "_f1" + ":" + str(f1score) + "\n")

            with open(result_filename, mode="a+") as f:
                f.writelines(results)

if __name__ == "__main__":
    graph_gan = GraphGAN()
    graph_gan.train()

