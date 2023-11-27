import networkx as nx
import pandas as pd


# get degree for each node
from matplotlib import pyplot as plt

dataset_dir = "../gold/mock"
df1 = pd.read_csv(f"{dataset_dir}/train1.txt", header=None, sep="\t", names=["h", "r", "t"])
df2 = pd.read_csv(f"{dataset_dir}/train2.txt", header=None, sep="\t", names=["h", "r", "t"])
df3 = pd.read_csv(f"{dataset_dir}/train.txt", header=None, sep="\t", names=["h", "r", "t"])
df4 = pd.read_csv(f"{dataset_dir}/valid.txt", header=None, sep="\t", names=["h", "r", "t"])
df = pd.concat([df1, df2, df3, df4], axis=0)

G = nx.DiGraph()
edges = [(row['h'], row['t'], {'relation': row['r']}) for _, row in df.iterrows()]
G.add_edges_from(edges)
degrees = dict(G.degree())
degrees_df = pd.DataFrame(list(degrees.items()), columns=['gene', 'degree'])

# mrr per node
mrr = pd.read_csv(f"{dataset_dir}/mrr.csv", header=0)
entity_vocab = pd.read_csv(f"{dataset_dir}/entity_vocab.csv", header=0)
mrr = pd.merge(mrr, entity_vocab, on="id", how="left")
mrr = pd.merge(mrr, degrees_df, on="gene", how="left")

# test nodes
test = pd.read_csv(f"{dataset_dir}/test_eval.txt", header=None, sep="\t", names=["h", "r", "t"])
test_genes = pd.concat([test["h"], test["t"]], axis=0).drop_duplicates()
test_genes = pd.DataFrame({'gene': test_genes})
df = pd.merge(mrr, test_genes, on="gene")

# scatter
plt.scatter(df['degree'], df['head_pred_mrr'], label='Head Pred MRR')
plt.scatter(df['degree'], df['tail_pred_mrr'], label='Tail Pred MRR')

plt.xlabel('Degree')
plt.ylabel('MRR')
plt.title('Degree vs MRR')
plt.legend()
plt.show()