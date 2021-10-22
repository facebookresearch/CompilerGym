# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
import sys

import hydra
from llvm_autotuning.experiment import Experiment
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from compiler_gym.util.shell_format import indent


@hydra.main(config_path="config", config_name="default")
def main(config: DictConfig) -> None:
    logging.basicConfig(level=logging.DEBUG)

    # Parse the config to pydantic models.
    OmegaConf.set_readonly(config, True)
    try:
        model: Experiment = Experiment(working_directory=os.getcwd(), **config)
    except ValidationError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    print("Experiment configuration:")
    print()
    print(indent(model.yaml()))
    print()

    model.run()
    print()
    print("Results written to", model.working_directory)


if __name__ == "__main__":
    main()
