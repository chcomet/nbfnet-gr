import re

import pandas as pd
from data.scripts import GTF

file_name = "lncRNA_gene_pairs"

# read dataframe
df = pd.read_csv(file_name, header=0, sep="\t", names=["idx", "t", "h"])
df = df[["h", "t"]]

# split list h
df_split = df['h'].str.split(';', expand=True).stack().reset_index(level=1, drop=True)
df = df.drop('h', axis=1).join(df_split.rename('h')).reset_index(drop=True)
df = df[df["h"] != ""]

# split list t
df_split = df['t'].str.split(';', expand=True).stack().reset_index(level=1, drop=True)
df = df.drop('t', axis=1).join(df_split.rename('t')).reset_index(drop=True)
df = df[df["t"] != ""]


# remove ensamble version
remove_version = lambda s: re.sub(r'\.\d+;?$', '', s)
df["h"] = df['h'].apply(remove_version)
df["t"] = df['t'].apply(remove_version)

# read ensamble database
homosapiens = GTF.dataframe("../Homo_sapiens.GRCh38.110.gtf")
homosapiens = homosapiens[["gene_id", "gene_name"]].drop_duplicates()

# join homosapiens to get gene name
df = pd.merge(df, homosapiens, left_on="h", right_on="gene_id", how="left")
df = pd.merge(df, homosapiens, left_on="t", right_on="gene_id", how="left")
df.rename(columns={'gene_name_x': 'h_name', 'gene_name_y': 't_name'}, inplace=True)

# read train, valid, test dataset
train = pd.read_csv("../../gold/lnctardppi_test/train2.txt", header=None, sep="\t", names=["head", "relation", "tail"])
test = pd.read_csv("../../gold/lnctardppi_test/test_pairs.txt", header=None, sep="\t", names=["head", "relation", "tail"])
valid = pd.read_csv("../../gold/lnctardppi_test/valid.txt", header=None, sep="\t", names=["head", "relation", "tail"])
train["split"] = "train"
test["split"] = "test"
valid["split"] = "valid"
lnctard = pd.concat([train, valid, test], axis=0, ignore_index=True)
nodes = pd.concat([lnctard["head"], lnctard["tail"]], axis=0, ignore_index=True).drop_duplicates()

# join to see whether in our DB in which split
df = pd.merge(df, lnctard, left_on=["h_name", "t_name"], right_on=["head", "tail"], how="left")
df["h_in_db"] = df['h_name'].isin(nodes)
df["t_in_db"] = df['t_name'].isin(nodes)

df = df[["h", "h_name", "h_in_db",  "t", "t_name", "t_in_db", "relation", "split"]].drop_duplicates()
df.to_csv(f"{file_name}.csv", index=False, header=True, sep="\t")