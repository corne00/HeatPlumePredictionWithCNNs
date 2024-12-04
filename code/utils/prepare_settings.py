import argparse
import datetime
import json
import yaml
import pathlib

def prepare_settings():
    now = datetime.datetime.now()
    current_time = now.strftime("%Y_%m_%d_%H_%M_%S_") + f"{now.microsecond // 1000:03d}"

    parser = argparse.ArgumentParser(description='Train a U-Net model with optional communication network.')
    # Set the save path to include a timestamp
    parser.add_argument('--save_path', type=str, default=rf"{current_time}",
                        help="Path for saving results (default: ./results/<timestamp>).")

    args = parser.parse_args()
    # STUDY_DIR = "/scratch/sgs/pelzerja/DDUNet/code/results/unittesting"
    # args.save_path = STUDY_DIR + "/energy_loss_finetune"
    save_path = pathlib.Path("./results") / args.save_path

    if save_path.exists():
        print(f"Path {save_path} already exists. Taking settings from there.")
        assert (save_path / "settings.yaml").exists(), f"No settings.yaml found in {save_path}"
        settings = yaml.safe_load(open(save_path / "settings.yaml"))

    else:
        settings = yaml.safe_load(open("default_settings.yaml"))
        # default_settings["save_path"] = STUDY_DIR + "/energy_loss_finetune"
        save_path.mkdir(parents=True, exist_ok=True)

    if settings["model"]["padding"] is None:
        settings["model"]["padding"] = settings["model"]["kernel_size"] // 2

    return settings, save_path


def save_args_to_json(args, filename="args.json"):
    # Convert the argparse.Namespace object to a dictionary
    args_dict = vars(args)
    
    # Save the dictionary to a JSON file
    with open(filename, 'w') as f:
        json.dump(args_dict, f, indent=4)

    print(f"Arguments saved to {filename}")

if __name__ == "__main__":
    args = prepare_settings()
    print(args)
