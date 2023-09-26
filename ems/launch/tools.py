# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import hydra
import os


def resolve_cfg_path(cfg: DictConfig):
    working_dir = os.getcwd()
    cfg.working_dir = working_dir
