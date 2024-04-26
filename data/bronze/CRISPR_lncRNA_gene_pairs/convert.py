import pandas as pd

train = pd.read_csv("../../gold/lnctardppi_test/train2.txt", header=None, sep="\t", names=["head", "relation", "tail"])
test = pd.read_csv("../../gold/lnctardppi_test/test_pairs.txt", header=None, sep="\t", names=["head", "relation", "tail"])
valid = pd.read_csv("../../gold/lnctardppi_test/valid.txt", header=None, sep="\t", names=["head", "relation", "tail"])

train["split"] = "train"
test["split"] = "test"
valid["split"] = "valid"

lnctard = pd.concat([train, valid, test], ignore_index=True)
nodes = pd.concat([lnctard["head"], lnctard["tail"]], axis=0, ignore_index=True).drop_duplicates()

lncrnas = [
    "CCAT1",
    "EPB41L4A-AS1",
    # "KB-1471A8.2",  not in LncTarD 2.0
    "LAMTOR5-AS1",
    # "LAMTOR5-AS1",  not in LncTarD 2.0
    "MIR17HG",
    "MIR210HG",
    "PVT1",
    "SNHG1",
    "SNHG12",
    "ZNF407-AS1"
]

crispr = pd.DataFrame([], columns=["h", "t"])

for rna in lncrnas:
    tmp_df = pd.DataFrame()
    file_path = f"{rna}.txt"
    with open(file_path, 'r') as file:
        t_values = file.read().splitlines()
    tmp_df["t_name"] = t_values
    tmp_df["h_name"] = rna
    crispr = pd.concat([crispr, tmp_df], ignore_index=True)


crispr = pd.merge(crispr, lnctard, left_on=["h_name", "t_name"], right_on=["head", "tail"], how="left")
crispr["h_in_db"] = crispr['h_name'].isin(nodes)
crispr["t_in_db"] = crispr['t_name'].isin(nodes)

crispr = crispr[["h_name", "h_in_db",  "t_name", "t_in_db", "relation", "split"]].drop_duplicates()
crispr.to_csv("crispr.csv", header=True, sep="\t", index=False)
