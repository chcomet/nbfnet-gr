import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def plot_graph(G):
    pos = nx.spring_layout(G)
    options = {"node_color": "black", "node_size": 1, "linewidths": 0, "width": 0.1}
    nx.draw(G, pos, **options)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='gray')
    plt.title("Graph")
    plt.axis('off')
    plt.show()


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


# create raw relation table
# lnctard = pd.read_csv("../bronze/lncTarD2.txt", sep="\t", header=0, encoding="latin-1")
# lnctard = lnctard[["Regulator", "SearchregulatoryMechanism", "Target"]].drop_duplicates()
# lnctard.to_csv("../silver/relations.csv", header=True, sep="\t", encoding="latin-1", index=False)

# manual clean up

# read dataframes
relations = pd.read_csv("../silver/relations.csv", sep="\t", header=0, encoding="latin-1")
genes = pd.read_csv("../silver/genes_matched.csv", sep="\t", header=0, encoding="latin-1")
genes = genes[["gene_name_corrected", "gene_name_lnctard"]]

# join on name
relations = pd.merge(relations, genes, how="left", left_on="Regulator", right_on="gene_name_lnctard")
relations = pd.merge(relations, genes, how="left", left_on="Target", right_on="gene_name_lnctard")
relations = relations[["gene_name_corrected_x", "SearchregulatoryMechanism", "gene_name_corrected_y"]].drop_duplicates()

# filter out unmatched genes
relations = relations[relations["gene_name_corrected_x"].notnull() & relations["gene_name_corrected_y"].notnull()]

# create graph
G = nx.DiGraph()
for _, row in relations.iterrows():
    G.add_edge(row['gene_name_corrected_x'], row['gene_name_corrected_y'], relation=row['SearchregulatoryMechanism'])
plot_cc(G)
plot_degree(G)

# extract largest connected component
largest_connected_component = max(nx.weakly_connected_components(G), key=len)
G_sub = G.subgraph(largest_connected_component)
plot_degree(G_sub)

# save largest connected component
edges = [(u, G_sub[u][v]['relation'], v) for u, v in G_sub.edges()]
result_df = pd.DataFrame(edges, columns=['gene_name_corrected_y', 'SearchregulatoryMechanism', 'gene_name_corrected_y'])
result_df.to_csv("../silver/relations_subgraph.csv", header=False, sep="\t", index=False)


