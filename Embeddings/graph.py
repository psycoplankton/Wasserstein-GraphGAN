import pandas as pd
import config
import networkx as nx

df = pd.read_csv(config.dataset,
                sep = '\t',
                names = ["NodeIDfrom", "NodeIDto"],
                )

G = nx.from_pandas_edgelist(df = df,
                             source = "NodeIDfrom",
                             target = "NodeIDto",
                             create_using=nx.Graph())