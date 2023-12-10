import dotenv
import hydra
from omegaconf import DictConfig
import argparse
import sys

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

# # load environment variables from `.env` file if it exists
# # recursively searches for `.env` in all folders starting from work dir
# dotenv.load_dotenv(override=True)

@hydra.main(version_base="1.2", config_path=root / "configs", config_name="train.yaml")
def main(config: DictConfig):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src import utils
    from src.training_pipeline import train

    # Applies optional utilities
    utils.extras(config)

    # Train model
    return train(config)


if __name__ == "__main__":
    sys.argv = [s for s in sys.argv if "--mode" not in s and "--port" not in s]
    main()
