# Guided Research: BioKGC Application of lncRNA Regulation

BioKGC is a graph neural network framework, adapted from [NBFNet](https://arxiv.org/pdf/2106.06935.pdf), designed to reason on biomedical knowledge graphs. BioKGC learns path representations (instead of commonly used node embeddings) for the task of link prediction, specifically taking into account different node types and a background regulatory graph for message passing.

To deepen the understanding of lncRNA-mediated regulation, this project constructed a knowledge graph of lncRNA regulation using the [LncTarD 2.0](https://lnctard.bio-database.com/) dataset and employed BioKGC to identify and infer potential novel regulatory instances, which may help to further reveal new insights and potential treatments for diseases through.

This codebase is based on [NBFNet][https://github.com/DeepGraphLearning/NBFNet] and [BioKGC](https://github.com/emyyue/NBFNet)

## Installation 

```bash
pip install torch-scatter torch-cluster -f https://pytorch-geometric.com/whl/torch-2.0.0+cu118.html
pip install torchdrug pykeen ogb easydict pyyaml
```

## Run

Here are sample commands to run the scripts:
```bash
python script/run.py -c config/knowledge_graph/lnctardppi.yaml --gpus [0] --version v1
python script/evaluate.py -c config/knowledge_graph/lnctardppi_eval.yaml --gpus [0] --checkpoint /root/nbfnet-gr/experiments/KnowledgeGraphCompletionBiomed/biomedical/NBFNet/lnctardppi/model_epoch_9.pth
python script/predict.py -c config/knowledge_graph/lnctardppi_pred.yaml --gpus [0] --checkpoint /root/nbfnet-gr/experiments/KnowledgeGraphCompletionBiomed/biomedical/NBFNet/lnctardppi/model_epoch_9.pth
python script/visualize_graph.py -c config/knowledge_graph/lnctardppi_vis.yaml --gpus [0] --checkpoint /root/nbfnet-gr/experiments/KnowledgeGraphCompletionBiomed/biomedical/NBFNet/lnctardppi/model_epoch_9.pth
```