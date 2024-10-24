import argparse
import datetime
import json

def parse_args():
    # Get the current date and time formatted as YYYY-DD-HH-MM
    # Get the current date and time formatted as YYYY_MM_DD_HH_MM_SS_MS (milliseconds)
    now = datetime.datetime.now()
    current_time = now.strftime("%Y_%m_%d_%H_%M_%S_") + f"{now.microsecond // 1000:03d}"

    # current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    
    parser = argparse.ArgumentParser(description='Set subdomain distribution and other training parameters.')

    parser.add_argument('--subdomains_dist', type=int, nargs=2, default=(1, 1),
                        help='Distribution of subdomains (default: (1, 1)).')
    parser.add_argument('--depth', type=int, default=4,
                        help='Depth of the encoder-decoder model (default: 4).')
    parser.add_argument('--complexity', type=int, default=8,
                        help='Complexity level (number of convolutions in the first layer) (default: 8).')
    parser.add_argument('--kernel_size', type=int, default=5,
                        help='Size of the used convolutional kernel (default: 5).')
    parser.add_argument('--padding', type=int, default=None,
                        help='Padding size (default: kernel_size // 2). If not specified, it will be set to kernel_size // 2.')
    parser.add_argument('--comm', type=bool, default=False,
                        help='Enable coarse network (default: False).')
    parser.add_argument('--num_comm_fmaps', type=int, default=0,
                        help='Number of feature maps sent to the coarse network (default: 0).')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs (default: 100).')
    parser.add_argument('--exchange_fmaps', type=bool, default=False,
                        help='Enable feature map exchange between subdomains (default: False).')
    parser.add_argument('--batch_size_training', type=int, default=16,
                        help='Batch size used for training (default: 16).')
    parser.add_argument('--batch_size_testing', type=int, default=16,
                        help='Batch size used for testing and validation (default: 16).')
    parser.add_argument('--num-convs', type=int, default=2,
                        help='Number of convolutions in each block (default: 2).')
    parser.add_argument('--val_loss', type=str, default="combi_0_75")
    
    # Set the save path to include a timestamp
    parser.add_argument('--save_path', type=str, default=rf"./results/{current_time}",
                        help="Path for saving results (default: ./results/<timestamp>).")

    args = parser.parse_args()

    # Set padding to kernel_size // 2 if not provided
    if args.padding is None:
        args.padding = args.kernel_size // 2

    return args

import json


def save_args_to_json(args, filename="args.json"):
    # Convert the argparse.Namespace object to a dictionary
    args_dict = vars(args)
    
    # Save the dictionary to a JSON file
    with open(filename, 'w') as f:
        json.dump(args_dict, f, indent=4)

    print(f"Arguments saved to {filename}")

if __name__ == "__main__":
    args = parse_args()
    print(args)
