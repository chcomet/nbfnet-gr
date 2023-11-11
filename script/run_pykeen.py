import os

import numpy as np
import pandas as pd
import pykeen
from pykeen.pipeline import pipeline
from pykeen.triples.leakage import Sealant
from pykeen.triples import TriplesFactory
from pykeen.hpo import hpo_pipeline
from optuna.samplers import GridSampler

from matplotlib import pyplot as plt

print(pykeen.env())

train_df = pd.read_csv("../data/gold/lnctardv1/train2.txt", delimiter='\t', header=None)
valid_df = pd.read_csv("../data/gold/lnctardv1/valid.txt", delimiter='\t', header=None)
test_df = pd.read_csv("../data/gold/lnctardv1/test.txt", delimiter='\t', header=None)

training = TriplesFactory.from_labeled_triples(train_df.to_numpy())
valid = TriplesFactory.from_labeled_triples(valid_df.to_numpy())
testing = TriplesFactory.from_labeled_triples(test_df.to_numpy())

res = pd.DataFrame(
    columns=[
        'model', 'num_negs', 'batch_size', 'learning_rate', 'num_epochs',
        "hits_at_1", "hits_at_3", "hits_at_10", "arithmetic_mean_rank", "inverse_harmonic_mean_rank"
    ]
)

for num_negs_per_pos in [16, 32, 128]:
    for batch_size in [128, 256]:
        for lr, num_epochs in [(0.001, 200), (0.0001, 300)]:
            transe = pipeline(
                training=training,
                validation=valid,
                testing=testing,
                model="RotatE",
                model_kwargs=dict(embedding_dim=512),
                training_kwargs=dict(use_tqdm_batch=False, num_epochs=num_epochs, batch_size=batch_size),
                evaluation_kwargs=dict(use_tqdm=False),
                optimizer_kwargs=dict(lr=lr),
                negative_sampler_kwargs=dict(num_negs_per_pos=num_negs_per_pos),
                random_seed=1,
                device="cuda",
            )
            metric = transe.metric_results.to_df()
            new_record = {
                'model': "RotatE",
                'num_negs': num_negs_per_pos,
                'batch_size': batch_size,
                'learning_rate': lr,
                'num_epochs': num_epochs,
                "hits_at_1": metric.loc[(metric.Side == "both") & (metric.Type == "realistic") & (metric.Metric == "hits_at_1"), "Value"].values[0],
                "hits_at_3": metric.loc[(metric.Side == "both") & (metric.Type == "realistic") & (metric.Metric == "hits_at_3"), "Value"].values[0],
                "hits_at_10": metric.loc[(metric.Side == "both") & (metric.Type == "realistic") & (metric.Metric == "hits_at_10"), "Value"].values[0],
                "arithmetic_mean_rank": metric.loc[(metric.Side == "both") & (metric.Type == "realistic") & (metric.Metric == "arithmetic_mean_rank"), "Value"].values[0],
                "inverse_harmonic_mean_rank": metric.loc[(metric.Side == "both") & (metric.Type == "realistic") & (metric.Metric == "inverse_harmonic_mean_rank"), "Value"].values[0]
            }
            print(new_record)
            res = res.append(new_record, ignore_index=True)

res.to_csv("/root/nbfnet-gr/experiments/pykeen_res.csv", sep="\t", index=False)