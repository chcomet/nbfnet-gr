import pandas as pd
import pykeen
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

print(pykeen.env())

train_df = pd.read_csv("/home/nbfnet-gr/data/gold/lnctardppi/train2.txt", delimiter='\t', header=None, names=["h", "r", "t"])
# ppi_df = pd.read_csv("/home/nbfnet-gr/data/gold/lnctardppi/train1.txt", delimiter='\t', header=None, names=["h", "r", "t"])
# train_df = pd.concat([train_df, ppi_df], axis=0).drop_duplicates()
valid_df = pd.read_csv("/home/nbfnet-gr/data/gold/lnctardppi/valid.txt", delimiter='\t', header=None, names=["h", "r", "t"])
test_df = pd.read_csv("/home/nbfnet-gr/data/gold/lnctardppi/test.txt", delimiter='\t', header=None, names=["h", "r", "t"])

training = TriplesFactory.from_labeled_triples(train_df.to_numpy(), create_inverse_triples=True)
valid = TriplesFactory.from_labeled_triples(valid_df.to_numpy(), create_inverse_triples=False)
testing = TriplesFactory.from_labeled_triples(test_df.to_numpy(), create_inverse_triples=False)

# df = pd.read_csv("/home/nbfnet-gt/data/bronze/lncTarD2.txt", sep="\t", encoding="latin-1", dtype="string")
# df = df[["Regulator", "SearchregulatoryMechanism", "Target"]]
# df = df[df["Regulator"].isin(df["Regulator"].value_counts().loc[lambda x: x > 1].index)].drop_duplicates().reset_index(drop=True).to_numpy()
# tf = TriplesFactory.from_labeled_triples(df)
# training, valid, testing = tf.split([0.8, 0.1, 0.1], random_state=1234)


res = pd.DataFrame({
    'model': pd.Series(dtype='str'),
    'num_negs': pd.Series(dtype='int'),
    'batch_size': pd.Series(dtype='int'),
    'learning_rate': pd.Series(dtype='float'),
    'num_epochs': pd.Series(dtype='int'),
    "hits@1": pd.Series(dtype='float'),
    "hits@3": pd.Series(dtype='float'),
    "hits@10": pd.Series(dtype='float'),
    "mr": pd.Series(dtype='float'),
    "mrr": pd.Series(dtype='float')
})

for num_negs_per_pos in [16, 32, 64, 128]:
    for batch_size in [64, 128, 256]:
        for lr, num_epochs in [(1e-02, 200), (1e-03, 300), (1e-04, 400)]:
            rotate = pipeline(
                training=training,
                validation=valid,
                testing=testing,
                model="RotatE",
                model_kwargs=dict(embedding_dim=256),
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
            res = pd.concat([res, pd.DataFrame(new_record, index=[0])], ignore_index=True)
            # rotate.save_to_directory(f'/root/nbfnet-gr/experiments/rotate/1024/neg{num_negs_per_pos}_bs{batch_size}_lr{lr}_e{num_epochs}')

res.to_csv("/root/nbfnet-gr/experiments/rotate_256.csv", sep="\t", index=False)