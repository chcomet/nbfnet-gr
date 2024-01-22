import pandas as pd
import pykeen
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

print(pykeen.env())

train_df = pd.read_csv("/home/nbfnet-gr/data/gold/lnctardppi/train2.txt", delimiter='\t', header=None, names=["h", "r", "t"])
valid_df = pd.read_csv("/home/nbfnet-gr/data/gold/lnctardppi/valid.txt", delimiter='\t', header=None, names=["h", "r", "t"])
test_df = pd.read_csv("/home/nbfnet-gr/data/gold/lnctardppi/test.txt", delimiter='\t', header=None, names=["h", "r", "t"])

training = TriplesFactory.from_labeled_triples(train_df.to_numpy(), create_inverse_triples=True)
valid = TriplesFactory.from_labeled_triples(valid_df.to_numpy(), create_inverse_triples=False)
testing = TriplesFactory.from_labeled_triples(test_df.to_numpy(), create_inverse_triples=False)

res = pd.DataFrame(
    columns=[
        'model', 'num_negs', 'batch_size', 'learning_rate', 'num_epochs',
        "hits@1", "hits@3", "hits@10", "mr", "mrr"
    ]
)

for num_negs_per_pos in [16, 32, 128]:
    for batch_size in [128, 256]:
        for lr, num_epochs in [(0.001, 200), (0.0001, 300)]:
            rotate = pipeline(
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
            metric = rotate.metric_results.to_df()
            new_record = {
                'model': "RotatE",
                'num_negs': num_negs_per_pos,
                'batch_size': batch_size,
                'learning_rate': lr,
                'num_epochs': num_epochs,
                "hits@1": metric.loc[(metric.Side == "both") & (metric.Type == "realistic") & (metric.Metric == "hits_at_1"), "Value"].values[0],
                "hits@3": metric.loc[(metric.Side == "both") & (metric.Type == "realistic") & (metric.Metric == "hits_at_3"), "Value"].values[0],
                "hits@10": metric.loc[(metric.Side == "both") & (metric.Type == "realistic") & (metric.Metric == "hits_at_10"), "Value"].values[0],
                "mr": metric.loc[(metric.Side == "both") & (metric.Type == "realistic") & (metric.Metric == "arithmetic_mean_rank"), "Value"].values[0],
                "mrr": metric.loc[(metric.Side == "both") & (metric.Type == "realistic") & (metric.Metric == "inverse_harmonic_mean_rank"), "Value"].values[0]
            }
            print(new_record)
            res = res.append(new_record, ignore_index=True)
            rotate.model.save_model(f'/root/nbfnet-gr/experiments/rotate/neg{num_negs_per_pos}_bs{batch_size}_lr{str(lr).split(".")[1]}_e{num_epochs}')



res.to_csv("/root/nbfnet-gr/experiments/pykeen_res.csv", sep="\t", index=False)