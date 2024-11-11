import torch
import optuna
import pathlib
import os

import hyperparam_optuna as hyp

def test_setup():
    STUDY_DIR = "/scratch/sgs/pelzerja/DDUNet/code/results/unittesting"
    study_dir = pathlib.Path(STUDY_DIR)
    study_dir.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(direction="minimize", storage=f"sqlite:///{STUDY_DIR}/hyperparam_opti.db", study_name="search", load_if_exists=True)
    study.optimize(hyp.objective, n_trials=2)

    os.system(f"rm -rf {STUDY_DIR}")

if __name__ == "__main__":
    test_setup()