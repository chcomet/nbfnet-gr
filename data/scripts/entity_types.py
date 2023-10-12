import pandas as pd


# load dataframes
relations = pd.read_csv("../silver/relations_subgraph.csv", sep="\t", header=None, encoding="latin-1")
relations.columns = ["Head", "Relation", "Tail"]
genes = pd.read_csv("../silver/genes_matched.csv", sep="\t", header=0, encoding="latin-1")

# extract related columns
used_genes = pd.concat([relations["Head"], relations["Tail"]], axis=0).drop_duplicates()
used_genes = pd.DataFrame({'gene': used_genes})

# join
genes = pd.merge(genes, used_genes, left_on="gene_name_corrected", right_on="gene", how="inner")
genes = genes[["gene_name_corrected", "gene_type_corrected"]].drop_duplicates()
genes.to_csv("../gold/lnctard2/entity_types.txt", sep="\t", header=None, index=False)

