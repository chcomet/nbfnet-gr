from collections import Counter

import pandas as pd
from matplotlib import pyplot as plt


# load dataset
lnctard_train = pd.read_csv("../gold/lnctardppi/train2.txt", header=None, sep="\t", names=["h", "r", "t"])
df1 = pd.read_csv("../gold/lnctardppi/test.txt", header=None, sep="\t", names=["query_node", "relation", "prediction_node"])
df2 = pd.read_csv("../gold/lnctardppi/valid.txt", header=None, sep="\t", names=["query_node", "relation", "prediction_node"])
lnctard_test = pd.concat([df1, df2], axis=0).drop_duplicates()
malat_as_head = lnctard_train[lnctard_train["h"] == "MALAT1"]
malat_as_tail = lnctard_train[lnctard_train["t"] == "MALAT1"]

# # plot malat_as_head relation distribution
# fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
# plt.suptitle("Relation Distributions of Training Data")
# counter = Counter(list(malat_as_head["r"]))
# axs[0].barh(list(counter.keys()), list(counter.values()))
# axs[0].set_title("MALAT1 as head")
# counter = Counter(list(malat_as_tail["r"]))
# axs[1].barh(list(counter.keys()), list(counter.values()))
# axs[1].set_title("MALAT1 as tail")
# plt.tight_layout()
# plt.show()


# load entity types
entity_types = pd.read_csv("../gold/lnctardppi/entity_types.txt", header=None, sep="\t",
                           names=["prediction_node", "prediction_node_type"])
# load predictions
preds = pd.read_csv("../gold/lnctardppi_pred/predictions_MALAT1.csv", header=0, sep="\t")
preds = preds[["query_node", "query_relation", "reverse", "prediction_node", "probability"]]
head_preds = preds[preds["reverse"] == 1]

# novel tail predictions
tail_preds = preds[preds["reverse"] == 0]
tail_preds = pd.merge(tail_preds, entity_types, on="prediction_node", how="left")
existing_tail_nodes = set(malat_as_head["t"].drop_duplicates())
existing_tail_preds = tail_preds[tail_preds["prediction_node"].isin(existing_tail_nodes)]
novel_tail_preds = tail_preds[~tail_preds["prediction_node"].isin(existing_tail_nodes)]
novel_tail_preds = novel_tail_preds[novel_tail_preds["prediction_node_type"] != "protein_coding_ppi"]
novel_tail_preds = pd.merge(novel_tail_preds, lnctard_test, on=["query_node", "prediction_node"], how="left")

# novel head predictions
existing_head_nodes = set(malat_as_tail["h"].drop_duplicates())
existing_head_preds = tail_preds[head_preds["prediction_node"].isin(existing_head_nodes)]
novel_head_preds = head_preds[~head_preds["prediction_node"].isin(existing_head_nodes)]
