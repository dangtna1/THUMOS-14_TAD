import os
import sys

sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)


import argparse

from mmengine.config import Config, DictAction

from tad.datasets import build_dataset, build_dataloader
from tad.datasets import transforms  # registers everything under transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Temporal Action Detector")
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--id", type=int, default=0, help="repeat experiment id")
    parser.add_argument(
        "--resume", type=str, default=None, help="resume from a checkpoint"
    )
    parser.add_argument(
        "--not_eval", action="store_true", help="whether not to eval, only do inference"
    )
    parser.add_argument(
        "--disable_deterministic",
        action="store_true",
        help="disable deterministic for faster speed",
    )
    parser.add_argument(
        "--cfg-options", nargs="+", action=DictAction, help="override settings"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    print(f"Loaded config from {args.config}")

    # # setup logger
    # logger = setup_logger("Train", save_dir=cfg.work_dir, distributed_rank=args.rank)
    # logger.info(f"Using torch version: {torch.__version__}, CUDA version: {torch.version.cuda}")
    # logger.info(f"Config: \n{cfg.pretty_text}")

    # build dataset
    train_dataset = build_dataset(cfg.dataset.train, default_args=dict(logger=None))
    # train_loader = build_dataloader(
    #     train_dataset,
    #     rank=args.rank,
    #     world_size=args.world_size,
    #     shuffle=True,
    #     drop_last=True,
    #     **cfg.solver.train,
    # )


if __name__ == "__main__":
    main()
