import os
import sys
import math
import pprint

import torch

from torchdrug import core, models
from torchdrug.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nbfnet import dataset, layer, model, task, util


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
    util.solver_load(solver, "/root/nbfnet-gr/experiments/KnowledgeGraphCompletionBiomed/biomedical/NBFNet/2023-11-06-00-10-27-809338/model_epoch_7.pth")

    test(cfg, solver)
