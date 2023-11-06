import os
import sys
import math
import pprint

import torch

from torchdrug import core, models
from torchdrug.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nbfnet import dataset, layer, model, task, util


def solver_load(checkpoint, load_optimizer=True):
    if comm.get_rank() == 0:
        logger.warning("Load checkpoint from %s" % checkpoint)
    checkpoint = os.path.expanduser(checkpoint)
    state = torch.load(checkpoint, map_location=solver.device)
    # some issues with loading back the fact_graph and graph
    # remove
    state["model"].pop("fact_graph")
    state["model"].pop("graph")
    state["model"].pop("undirected_fact_graph")
    # load without
    solver.model.load_state_dict(state["model"], strict=False)

    if load_optimizer:
        solver.optimizer.load_state_dict(state["optimizer"])
        for state in solver.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(solver.device)

    comm.synchronize()


def test(cfg, solver):
    solver.model.split = "valid"
    solver.evaluate("valid")
    solver.model.split = "test"
    solver.evaluate("test")


if __name__ == "__main__":
    args, vars = util.parse_args()
    print(f"{args=}")
    print(f"{vars=}")
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + comm.get_rank())

    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))

    dataset = core.Configurable.load_config_dict(cfg.dataset)
    solver = util.build_solver(cfg, dataset)
    solver_load("/root/nbfnet-gr/experiments/KnowledgeGraphCompletionBiomed/biomedical/NBFNet/2023-11-06-00-10-27-809338/model_epoch_7.pth")

    test(cfg, solver)
