import os
import sys

sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)

import argparse
import torch
from mmengine.config import Config, DictAction

from tad.datasets import build_dataset, build_dataloader
from tad.models import build_detector
from tad.cores import (
    train_one_epoch,
    build_optimizer,
    build_scheduler,
)
from tad.utils import (
    setup_logger,
    ModelEma,
    save_checkpoint,
)


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

    # setup logger
    logger = setup_logger("Train", save_dir=cfg.work_dir)
    logger.info(
        f"Using torch version: {torch.__version__}, CUDA version: {torch.version.cuda}"
    )
    logger.info(f"Config: \n{cfg.pretty_text}")

    # build dataset
    train_dataset = build_dataset(cfg.dataset.train, default_args=dict(logger=logger))
    train_loader = build_dataloader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        **cfg.solver.train,
    )

    # Debug
    # batch = next(iter(train_loader))
    # print(batch.keys())
    # print(batch["inputs"].shape)
    # print(batch["masks"].shape)
    # print(batch["gt_segments"])
    # print(batch["gt_labels"])
    # print(batch["metas"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model
    model = build_detector(cfg.model).to(device)

    # Model EMA
    use_ema = getattr(cfg.solver, "ema", False)
    if use_ema:
        model_ema = ModelEma(model)
    else:
        model_ema = None

    # build optimizer and scheduler
    optimizer = build_optimizer(cfg.optimizer, model, logger=logger)
    scheduler, max_epoch = build_scheduler(cfg.scheduler, optimizer, len(train_loader))

    # override the max_epoch
    # max_epoch = cfg.workflow.get("end_epoch", max_epoch)

    # train the detector
    logger.info("Training Starts...\n")
    logger.info(max_epoch)
    # val_loss_best = 1e6
    # val_start_epoch = cfg.workflow.get("val_start_epoch", 0)
    for epoch in range(0, max_epoch):
        # train for one epoch
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            logger=logger,
            model_ema=model_ema,
            clip_grad_l2norm=cfg.solver.clip_grad_norm,
            logging_interval=cfg.workflow.logging_interval,
            scaler=None,
        )

        # save checkpoint
        if (epoch == max_epoch - 1) or (
            (epoch + 1) % cfg.workflow.checkpoint_interval == 0
        ):
            save_checkpoint(
                model, model_ema, optimizer, scheduler, epoch, work_dir=cfg.work_dir
            )

        # val for one epoch
        # if epoch >= val_start_epoch:
        #     if (cfg.workflow.val_loss_interval > 0) and (
        #         (epoch + 1) % cfg.workflow.val_loss_interval == 0
        #     ):
        #         val_loss = val_one_epoch(
        #             val_loader,
        #             model,
        #             logger,
        #             0,
        #             epoch,
        #             model_ema=model_ema,
        #             use_amp=use_amp,
        #         )
        #
        #         # save the best checkpoint
        #         if val_loss < val_loss_best:
        #             logger.info(f"New best epoch {epoch}")
        #             val_loss_best = val_loss
        #             if 0 == 0:
        #                 save_best_checkpoint(
        #                     model, model_ema, epoch, work_dir=cfg.work_dir
        #                 )

        # eval for one epoch
        # if epoch >= val_start_epoch:
        #     if (cfg.workflow.val_eval_interval > 0) and (
        #         (epoch + 1) % cfg.workflow.val_eval_interval == 0
        #     ):
        #         eval_one_epoch(
        #             test_loader,
        #             model,
        #             cfg,
        #             logger,
        #             args.rank,
        #             model_ema=model_ema,
        #             use_amp=use_amp,
        #             world_size=args.world_size,
        #             not_eval=args.not_eval,
        #         )


if __name__ == "__main__":
    main()
