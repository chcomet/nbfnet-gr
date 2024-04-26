import pandas as pd
import pykeen
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory


print(pykeen.env())

res = pd.DataFrame(
    columns=[
        'model', 'embedding_dim', 'batch_size',
        "hits_at_1", "hits_at_3", "hits_at_10", "mr", "mrr"
    ]
)

training = TriplesFactory.from_path("data/gold/lnctardppi/train2.txt", delimiter='\t', header=None)
validation = TriplesFactory.from_path("data/gold/lnctardppi/valid.txt", delimiter='\t', header=None)
testing = TriplesFactory.from_path("data/gold/lnctardppi/test.txt", delimiter='\t', header=None)

for model in ["TransE", "DistMult"]:
    for embedding_dim in [256, 512, 1024]:
        for batch_size in [128, 256, 512]:
            transe = pipeline(
                training=training,
                validation=validation,
                testing=testing,
                model=model,
                model_kwargs=dict(embedding_dim=embedding_dim),
                training_kwargs=dict(use_tqdm_batch=False, num_epochs=500, batch_size=batch_size),
                evaluation_kwargs=dict(use_tqdm=False),
                optimizer_kwargs=dict(lr=0.001),
                negative_sampler_kwargs=dict(num_negs_per_pos=32),
                random_seed=1,
                device="cuda",
                stopper='early',
                stopper_kwargs=dict(patience=10, relative_delta=0.002),
            )
            metric = transe.metric_results.to_df()
            new_row = {
                'model': model,
                'embedding_dim': embedding_dim,
                'batch_size': batch_size,
                "hits_at_1": metric.loc[(metric.Side == "both") & (metric.Type == "realistic") & (metric.Metric == "hits_at_1"), "Value"].values[0],
                "hits_at_3": metric.loc[(metric.Side == "both") & (metric.Type == "realistic") & (metric.Metric == "hits_at_3"), "Value"].values[0],
                "hits_at_10": metric.loc[(metric.Side == "both") & (metric.Type == "realistic") & (metric.Metric == "hits_at_10"), "Value"].values[0],
                "mr": metric.loc[(metric.Side == "both") & (metric.Type == "realistic") & (metric.Metric == "arithmetic_mean_rank"), "Value"].values[0],
                "mrr": metric.loc[(metric.Side == "both") & (metric.Type == "realistic") & (metric.Metric == "inverse_harmonic_mean_rank"), "Value"].values[0]
            }
            print(new_row)
            res = pd.concat([res, pd.DataFrame([new_row])], ignore_index=True)

res.to_csv("/root/nbfnet-gr/pykeen_hypo.csv", sep="\t", index=False)