import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import GTF


def plot_degree(G):
    node_degrees = dict(G.degree())
    degree_values = list(node_degrees.values())
    plt.hist(degree_values, bins=range(min(degree_values), max(degree_values) + 1), alpha=0.7)
    plt.xlabel('Degree')
    plt.ylabel('Count')
    plt.title('Degree Distribution')
    plt.show()


def plot_cc(G):
    connected_components = list(nx.strongly_connected_components(G))
    component_sizes = [len(component) for component in connected_components]  # 1, 2, 282
    print(set(component_sizes))
    plt.hist(component_sizes, bins=range(min(component_sizes), max(component_sizes) + 1), alpha=0.7)
    plt.xlabel('Component Size')
    plt.ylabel('Count')
    plt.title('Connected Component Size Distribution')
    plt.show()


# load ppi
ppi = pd.read_csv("../bronze/ppi.txt", sep=",", header=0)

# appeared gene
ppi_genes = pd.concat([ppi["h"], ppi["t"]], axis=0).drop_duplicates()
ppi_genes = pd.DataFrame({'gene_name': ppi_genes})

# load homosapiens
homosapiens = GTF.dataframe("../bronze/Homo_sapiens.GRCh38.110.gtf")
homosapiens = homosapiens[["gene_id", "gene_name", "gene_biotype"]].drop_duplicates()

# find the protein coding genes
ppi_proteins = pd.merge(ppi_genes, homosapiens, on="gene_name", how="left")
ppi_proteins = ppi_proteins[ppi_proteins["gene_biotype"] == "protein_coding"]
ppi_proteins = ppi_proteins[["gene_name"]]

# drop other genes of other gene types
ppi = pd.merge(ppi, ppi_proteins, left_on="h", right_on="gene_name", how="inner")
ppi = pd.merge(ppi, ppi_proteins, left_on="t", right_on="gene_name", how="inner")
ppi = ppi[["h", "r", "t"]]

# create graph
G = nx.DiGraph()
for _, row in ppi.iterrows():
    G.add_edge(row['h'], row['t'], relation=row['r'])
plot_cc(G)
plot_degree(G)

# extract largest connected component
largest_connected_component = max(nx.strongly_connected_components(G), key=len)
G_sub = G.subgraph(largest_connected_component)
plot_degree(G_sub)

# save largest connected component
edges = [(u, G_sub[u][v]['relation'], v) for u, v in G_sub.edges()]
result_df = pd.DataFrame(edges, columns=['h', 'r', 't'])
result_df.to_csv("../gold/lnctardppi/train2.txt", header=False, sep="\t", index=False)

# generate entity type
ppi = pd.read_csv("../gold/lnctardppi/train2.txt", header=None, sep="\t", names=["h", "r", "t"])
ppi_genes = pd.concat([ppi["h"], ppi["t"]], axis=0).drop_duplicates()
ppi_genes = pd.DataFrame({'gene': ppi_genes})
types = pd.read_csv("../gold/lnctard/entity_types.txt", header=None, sep="\t", names=["gene", "type"])
ppi_genes = ppi_genes[~ppi_genes["gene"].isin(types["gene"])]
ppi_genes["type"] = "protein_coding"
types = pd.concat([types, ppi_genes], axis=0).drop_duplicates()
types.to_csv("../gold/lnctardppi/entity_types.txt", header=False, sep="\t", index=False)

# generate entity name
types = pd.read_csv("../gold/lnctardppi/entity_types.txt", header=None, sep="\t", names=["gene", "type"])
names = pd.concat([types["gene"], types["gene"]], axis=1)
names.to_csv("../gold/lnctardppi/entity_names.txt", header=False, sep="\t", index=False)