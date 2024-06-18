import Walker
import gensim
import config
from graph import G
from Embeddings.utils import get_embeddings, write_embeddings

class DeepWalk:
    def __init__(self, graph, walk_length, num_walks, workers=1):

        self.graph = graph
        self.w2v_model = None
        self._embeddings = {}

        self.walker = Walker.RandomWalker(
            graph, p=1, q=1, )
        self.sentences = self.walker.simulate_walks(
            num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=1)

    def train(self, embed_size=128, window_size=5, workers=3, iter=5, **kwargs):

        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["vector_size"] = embed_size
        kwargs["sg"] = 1  # skip gram
        kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["epochs"] = iter

        print("Learning embedding vectors...")
        model = gensim.models.Word2Vec(**kwargs)
        print("Learning embedding vectors done!")

        self.w2v_model = model
        return model

deepwalk = DeepWalk(G, config.walk_length, config.num_walks) 
model = gensim.models.Word2Vec(sentences=deepwalk.sentences,
                 vector_size=config.vector_size,
                 epochs=config.epochs,
                 window = config.window_size,
                 compute_loss=True,
                 sg=1,
                 hs=0)

embeddings = get_embeddings(G, model)
write_embeddings(config.embeddings_filename, embeddings, G)