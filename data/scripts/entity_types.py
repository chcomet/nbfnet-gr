import pandas as pd


# load dataframes
lnctard1 = pd.read_csv("../gold/lnctardv1/train2.txt", header=None, sep="\t", names=["h", "r", "t"])
lnctard2 = pd.read_csv("../gold/lnctardv1/test.txt", header=None, sep="\t", names=["h", "r", "t"])
lnctard3 = pd.read_csv("../gold/lnctardv1/valid.txt", header=None, sep="\t", names=["h", "r", "t"])
lnctard = pd.concat([lnctard1, lnctard2, lnctard3], axis=0)
ppi = pd.read_csv("../gold/lnctardv1/train1.txt", header=None, sep="\t", names=["h", "r", "t"])
genes = pd.read_csv("../silver/genes_matched.csv", sep="\t", header=0, encoding="latin-1")

# extract related columns
lnctard_genes = pd.concat([lnctard["h"], lnctard["t"]], axis=0).drop_duplicates()
lnctard_genes = pd.DataFrame({'gene': lnctard_genes})
ppi_genes = pd.concat([ppi["h"], ppi["t"]], axis=0).drop_duplicates()
ppi_genes = pd.DataFrame({'gene': ppi_genes})

# get correct gene name and gene type of LncTarD
lnctard_type = pd.merge(genes, lnctard_genes, left_on="gene_name_corrected", right_on="gene", how="inner")
lnctard_type = lnctard_type[["gene_name_corrected", "gene_type_corrected"]].drop_duplicates()

# creat new gene type for genes from PPI
ppi_type = ppi_genes.merge(lnctard_genes, on="gene", how='left', indicator=True).query('_merge == "left_only"').drop(columns='_merge')
ppi_type.rename(columns={'gene': 'gene_name_corrected'}, inplace=True)
ppi_type["gene_type_corrected"] = "protein_coding_ppi"

types = pd.concat([lnctard_type, ppi_type], axis=0)
types.to_csv("../gold/lnctardv1/entity_types.txt", header=False, sep="\t", index=False)

names = pd.concat([types["gene_name_corrected"], types["gene_name_corrected"]], axis=1)
names.to_csv("../gold/lnctardv1/entity_names.txt", header=False, sep="\t", index=False)
