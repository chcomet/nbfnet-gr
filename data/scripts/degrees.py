import pandas as pd
import networkx as nx

df1 = pd.read_csv("../gold/lnctardppi/train1.txt", header=None, sep="\t", names=["h", "r", "t"])
df2 = pd.read_csv("../gold/lnctardppi/train2.txt", header=None, sep="\t", names=["h", "r", "t"])
df3 = pd.read_csv("../gold/lnctardppi/test.txt", header=None, sep="\t", names=["h", "r", "t"])
df4 = pd.read_csv("../gold/lnctardppi/valid.txt", header=None, sep="\t", names=["h", "r", "t"])
lnctard = pd.concat([df2, df3, df4], axis=0).drop_duplicates()
lnctard_ppi = pd.concat([df1, df2, df3, df4], axis=0).drop_duplicates()


h_degrees = pd.read_csv("degrees.csv", header=None, names=["h","h_type", "h_degree", "h_degree_ppi"])
t_degrees = pd.read_csv("degrees.csv", header=None, names=["t", "t_type", "t_degree", "t_degree_ppi"])

# entity_types = pd.read_csv("../gold/lnctardppi/entity_types.txt", header=None, sep="\t",
#                            names=["gene", "gene_type"])
#
#
# def get_degrees(df, col_name):
#     G = nx.DiGraph()
#     for _, row in df.iterrows():
#         G.add_edge(row['h'], row['t'], relation=row['r'])
#     return pd.DataFrame(
#         [(n, G.degree[n]) for n in G.nodes()],
#         columns=['gene', col_name]
#     )
#
#
# lnctard_degrees = get_degrees(lnctard, "degree_lnctard")
# lnctard_ppi_degrees = get_degrees(lnctard_ppi, "degree_lnctard_ppi")
# degrees = pd.merge(lnctard_degrees, lnctard_ppi_degrees, on="gene", how="left")
# degrees = pd.merge(degrees, entity_types, on="gene", how="left")
# degrees = degrees[["gene", "gene_type", "degree_lnctard", "degree_lnctard_ppi"]]
# degrees = degrees.sort_values(by=["degree_lnctard", "degree_lnctard_ppi"], ascending=False)
# degrees.to_csv("degrees.csv", index=False)


lnctard = pd.merge(lnctard, h_degrees, on="h", how="left")
lnctard = lnctard[lnctard["h_type"] == "lncRNA"]
lnctard = pd.merge(lnctard, t_degrees, on="t", how="left")
lnctard = lnctard[["h", "h_type", "h_degree", "r", "t", "t_type", "t_degree"]]
lnctard = lnctard.sort_values(by=["h_degree"], ascending=False)
lnctard.to_csv("lncRNA_degrees_with_PPI.csv", index=False)