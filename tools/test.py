import os
import sys

sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..")
print(path)
if path not in sys.path:
    sys.path.insert(0, path)

import argparse
import torch
from mmengine.config import Config, DictAction
from tad.models import build_detector
from tad.datasets import build_dataset, build_dataloader
from tad.cores import eval_one_epoch
from tad.utils import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Test a Temporal Action Detector")
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    parser.add_argument(
        "--checkpoint", type=str, default="none", help="the checkpoint path"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--id", type=int, default=0, help="repeat experiment id")
    parser.add_argument(
        "--not_eval",
        action="store_true",
        help="whether to not to eval, only do inference",
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
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # create_folder(cfg.work_dir)

    # setup logger
    logger = setup_logger("Test", save_dir=cfg.work_dir)
    logger.info(
        f"Using torch version: {torch.__version__}, CUDA version: {torch.version.cuda}"
    )
    logger.info(f"Config: \n{cfg.pretty_text}")

    # build dataset
    test_dataset = build_dataset(cfg.dataset.train, default_args=dict(logger=logger)) # change to test for real testing
    test_loader = build_dataloader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        **cfg.solver.test,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model
    model = build_detector(cfg.model).to(device)

    if (
        cfg.inference.load_from_raw_predictions
    ):  # if load with saved predictions, no need to load checkpoint
        logger.info(f"Loading from raw predictions: {cfg.inference.fuse_list}")
    else:  # load checkpoint: args -> config -> best
        if args.checkpoint != "none":
            checkpoint_path = args.checkpoint
        elif "test_epoch" in cfg.inference.keys():
            checkpoint_path = os.path.join(
                cfg.work_dir, f"checkpoint/epoch_{cfg.inference.test_epoch}.pth"
            )
        else:
            checkpoint_path = os.path.join(cfg.work_dir, "checkpoint/best.pth")
        logger.info("Loading checkpoint from: {}".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        logger.info("Checkpoint is epoch {}.".format(checkpoint["epoch"]))

        # Model EMA
        use_ema = getattr(cfg.solver, "ema", False)
        if use_ema:
            model.load_state_dict(checkpoint["state_dict_ema"])
            logger.info("Using Model EMA...")
        else:
            model.load_state_dict(checkpoint["state_dict"])

    # test the detector
    logger.info("Testing Starts...\n")
    eval_one_epoch(
        test_loader,
        model,
        cfg,
        logger,
        0,
        model_ema=None,  # since we have loaded the ema model above
        use_amp=False,
        world_size=0,
        not_eval=True,
    )
    logger.info("Testing Over...\n")


if __name__ == "__main__":
    main()
