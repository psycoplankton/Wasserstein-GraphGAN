from gensim.models import Word2Vec
from Walker import RandomWalker 
from graph import G
import config 
from Embeddings.utils import get_embeddings, write_embeddings
class Node2Vec:

    def __init__(self, graph, walk_length, num_walks, workers=1, use_rejection_sampling = False):

        self.graph = graph
        self.walker = RandomWalker(G)

        print("Preprocess transition probs...")
        self.walker.preprocess_transition_probs()

        self.sentences = self.walker.simulate_walks(
            num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=1)

    def train(self, embed_size=128, window_size=5, workers=3, iter=5, **kwargs):
        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["vector_size"] = embed_size
        kwargs["sg"] = 1
        kwargs["hs"] = 0  # node2vec not use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["epochs"] = iter

        print("Learning embedding vectors...")
        model = Word2Vec(**kwargs)
        print("Learning embedding vectors done!")

        self.w2v_model = model

        return model
    
model = Node2Vec(graph = G, walk_length = config.walk_length, num_walks = config.num_walks)
model.train(embed_size=config.vector_size, iter = config.epochs, window_size=config.window_size)
embeddings = get_embeddings(G, model)
write_embeddings(config.embeddings_filename, embeddings, G)
