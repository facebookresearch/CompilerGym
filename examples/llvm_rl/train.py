# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys

import hydra
from hydra.core.hydra_config import HydraConfig
from llvm_rl.model import Model
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import MissingMandatoryValue
from pydantic import ValidationError

import compiler_gym


def _get_job_id() -> int:
    try:
        return HydraConfig.get().job.id
    except MissingMandatoryValue:
        # The numeric job ID is missing if not in a multirun context. In that
        # case, there can only be a single run.
        return 0


@hydra.main(config_path="config", config_name="default")
def main(config: DictConfig) -> None:
    OmegaConf.set_readonly(config, True)

    # Parse the config to pydantic models.
    try:
        model: Model = Model(
            # Hydra changes the working directory.
            working_directory=os.getcwd(),
            job_id=_get_job_id(),
            compiler_gym_version=compiler_gym.__version__,
            **config
        )
    except ValidationError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    model.train()
    model.test()


if __name__ == "__main__":
    main()
