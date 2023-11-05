import pandas as pd


v1 = pd.read_csv("../bronze/lncTarD.txt", sep="\t", header=0, encoding="latin-1")
v1_relations = v1[["Regulator", "SearchregulatoryMechanism", "Target"]].drop_duplicates()
v1_genes = pd.concat([v1["Regulator"], v1["Target"]], axis=0).drop_duplicates()
v1_genes = pd.DataFrame({'gene': v1_genes})

v2 = pd.read_csv("../bronze/lncTarD2.txt", sep="\t", header=0, encoding="latin-1")
v2_relations = v2[["Regulator", "SearchregulatoryMechanism", "Target"]].drop_duplicates()
v2_genes = pd.concat([v2["Regulator"], v2["Target"]], axis=0).drop_duplicates()
v2_genes = pd.DataFrame({'gene': v2_genes})

# Gene Overlap
v1_genes_inner = pd.merge(v1_genes, v2_genes, how="inner")
v1_genes_anti = v1_genes[~v1_genes["gene"].isin(v1_genes_inner["gene"])]
print(f"There are {len(v1_genes)} nodes in LncTarD version 1.",
      f"{len(v1_genes_inner)} of them are also in LncTarD version 2. ")

# Relation Overlap
v1_relations_inner = pd.merge(v1_relations, v2_relations, on=["Regulator", "SearchregulatoryMechanism", "Target"], how="inner").drop_duplicates()
v1_relations_anti = (
    v1_relations
    .merge(v2_relations, on=["Regulator", "SearchregulatoryMechanism", "Target"], how='left', indicator=True)
    .query('_merge == "left_only"')
    .drop(columns=['_merge'])
)
print(f"There are {len(v1_relations)} edges in LncTarD version 1.",
      f"{len(v1_relations_inner)} of them are also in LncTarD version 2. ",
      f"{len(v1_relations_anti)} of them are unique.")


# Gene Name Correction
genes = pd.read_csv("../silver/genes_matched.csv", sep="\t", header=0, encoding="latin-1")
genes = genes[["gene_name_corrected", "gene_name_lnctard"]]

relation_inner_corrected = pd.merge(v1_relations_inner, genes, how="left", left_on="Regulator", right_on="gene_name_lnctard")
relation_inner_corrected = pd.merge(relation_inner_corrected, genes, how="left", left_on="Target", right_on="gene_name_lnctard")
relation_inner_corrected = relation_inner_corrected[["gene_name_corrected_x", "SearchregulatoryMechanism", "gene_name_corrected_y"]].drop_duplicates()
relation_inner_corrected = relation_inner_corrected[(relation_inner_corrected["gene_name_corrected_x"].notnull()) & (relation_inner_corrected["gene_name_corrected_y"].notnull())]
relation_inner_corrected.to_csv("../silver/relations_v1.csv", header=False, sep="\t", index=False)

relation_anti_corrected = pd.merge(v1_relations_anti, genes, how="left", left_on="Regulator", right_on="gene_name_lnctard")
relation_anti_corrected = pd.merge(relation_anti_corrected, genes, how="left", left_on="Target", right_on="gene_name_lnctard")
# relation_anti_corrected = relation_anti_corrected[["gene_name_corrected_x", "SearchregulatoryMechanism", "gene_name_corrected_y"]].drop_duplicates()
relation_anti_corrected.to_csv("../silver/relations_v1_problem.csv", header=True, sep="\t", index=False)