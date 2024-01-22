import pandas as pd
from matplotlib import pyplot as plt


# get degree for each node

path = "../gold/lnctardppi_pred"
degrees_df = pd.read_csv(f"{path}/degrees.csv", header=None, names=["gene", "gene_type", "degree", "degree_ppi"])

# mrr per node
mrr = pd.read_csv(f"{path}/mrr_per_node.csv", header=0)
entity_vocab = pd.read_csv(f"{path}/entity_vocab.csv", header=0)
mrr = pd.merge(mrr, entity_vocab, on="id", how="left")
mrr = pd.merge(mrr, degrees_df, on="gene", how="left")
mrr.to_csv(f"{path}/mrrs.csv", index=False)

# test nodes
test_genes = [
    "NEAT1", "MALAT1", "SNHG16", "ZFAS1", "MIR22HG", "LINC-PINT", "MIR4435-2HG",
    "FGD5-AS1", "CYTOR", "PVT1", "FTX", "FOXCUT", "FENDRR", "LINC00461", "DLEU2",
    "MIR34AHG", "LINC00511", "LINC02582", "DLGAP1-AS1", "THORLNC"
]
mrr_target = mrr[mrr["gene"].isin(test_genes)]

# mrr_tail = mrr_target[~mrr_target["tail_pred_mrr"].isnull()].reset_index()
# mrr_head = mrr_target[~mrr_target["head_pred_mrr"].isnull()].reset_index()
mrr_tail = mrr[~mrr["tail_pred_mrr"].isnull()].reset_index()
mrr_head = mrr[~mrr["head_pred_mrr"].isnull()].reset_index()

# scatter
plt.scatter(mrr_head['degree'], mrr_head['head_pred_mrr'], label='Head Pred MRR')
plt.scatter(mrr_tail['degree'], mrr_tail['tail_pred_mrr'], label='Tail Pred MRR')
# for i in range(len(mrr_tail)):
#     plt.text(mrr_tail['degree'][i], mrr_tail['tail_pred_mrr'][i], mrr_tail['gene'][i], fontsize=12, ha='right')

plt.xlabel('Degree')
plt.ylabel('MRR')
plt.title('Degree vs MRR')
plt.legend()
plt.show()