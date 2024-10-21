import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import sys

from models import *

# Set subdomains distribution  and other training parameters
subdomains_dist = (1,1)
domain_size = 1200
subdomain_dist = subdomains_dist
depth = 4
complexity = 32
save_path = "./results_new_data/baseline_1x1_with_comm/"
kernel_size = 5
padding = kernel_size // 2
comm = False
num_comm_fmaps = 256
num_epochs = 10000
exchange_fmaps = False
devices = ["cpu"]
factor = 1e10 # for plotting

def get_list_index(i, j, nx=subdomain_dist[0], ny=subdomain_dist[1]):
        return i * ny + j

def concatenate_tensors(tensors, nx=subdomain_dist[0], ny=subdomain_dist[1]):
    concatenated_tensors = []
    for i in range(nx):
        column_tensors = []
        for j in range(ny):
            index = get_list_index(i, j)
            column_tensors.append(tensors[index])
        concatenated_row = torch.cat(column_tensors, dim=2)
        concatenated_tensors.append(concatenated_row)

    result = torch.cat(concatenated_tensors, dim=3)
    return result


def plot_receptive_field(model, input_shape, target_position=None):
    model.eval()  # Set model to evaluation mode
    
    # Create a dummy input
    input_tensor = [torch.randn(1, *input_shape, requires_grad=True) for _ in range(subdomain_dist[0]*subdomain_dist[1])]
    
    # Forward pass through the model to get the output
    output = model(input_tensor)
    
    print(output.shape)
    if target_position is None:
        # Default target position is the center of the output
        target_position = (output.size(2) // 2, output.size(3) // 2)
    
    # Create a gradient tensor with the same shape as the output
    grad_output = torch.zeros_like(output)
    
    # Set the gradient at the target position to 1
    grad_output[0, 0, target_position[0], target_position[1]] = 1.0
    
    # Perform backward pass to get the gradient with respect to the input
    model.zero_grad()
    output.backward(grad_output)
    
    # Get the gradient with respect to the input

    # grad_input_concat = concatenate_tensors(input_tensor)
    # grad_input_concat.retain_grad()
    # print(grad_input_concat.grad)
    # grad_input = grad_input_concat.grad[0,0].detach().numpy()
    
    print("1", input_tensor[0].grad.shape)
    grad_input = concatenate_tensors([t.grad[:,:] for t in input_tensor])
    print("2", grad_input.shape)
    grad_input = grad_input[0,0].detach().numpy()

    print("Non zero values:", np.count_nonzero(grad_input))
    print("Square:", np.sqrt(np.count_nonzero(grad_input)))
    print()

    # Plot the resulting gradient
    plt.imshow(np.abs(grad_input), cmap='hot', interpolation='nearest', vmax=np.max(grad_input / factor))
    plt.colorbar()
    plt.title(f'Receptive Field at Output Position {target_position}')
    plt.savefig("receptive_field.png", bbox_inches='tight')
    # plt.show()

sys.path.append("../")
from models import MultiGPU_UNet_with_comm
# model = MultiGPU_UNet_with_comm(n_channels=3, n_classes=1, input_shape=(2560, 2560), num_comm_fmaps=num_comm_fmaps, devices=devices, depth=depth,
#                                subdom_dist=subdomain_dist, bilinear=False, comm=comm, complexity=complexity, dropout_rate=0.0, kernel_size=kernel_size, 
#                                padding=padding, communicator_type=None, comm_network_but_no_communication=(not exchange_fmaps), 
#                                communication_network_def=CNNCommunicatorDilated)

model = MultiGPU_UNet_with_comm(n_channels=3, n_classes=1, input_shape=(2560, 2560), num_comm_fmaps=num_comm_fmaps, devices=devices, depth=depth,
                                subdom_dist=subdomain_dist, bilinear=False, comm=comm, complexity=complexity, dropout_rate=0.0, kernel_size=kernel_size, 
                                padding=padding, communicator_type=None, comm_network_but_no_communication=(not exchange_fmaps), 
                                communication_network_def=CNNCommunicatorDilated)


input_shape = (3, domain_size // subdomains_dist[0], domain_size // subdomains_dist[1])  # Example input shape (channels, height, width)
target_layer_idx = 0  # The index of the target layer in the model (e.g., the first layer in the encoder)

plot_receptive_field(model, input_shape)

# Example usage
if __name__ == "__main__":
    sys.path.append("../")
    from models import MultiGPU_UNet_with_comm, CNNCommunicatorDilated
    # model = MultiGPU_UNet_with_comm(n_channels=3, n_classes=1, input_shape=(2560, 2560), num_comm_fmaps=num_comm_fmaps, devices=devices, depth=depth,
    #                                subdom_dist=subdomain_dist, bilinear=False, comm=comm, complexity=complexity, dropout_rate=0.0, kernel_size=kernel_size, 
    #                                padding=padding, communicator_type=None, comm_network_but_no_communication=(not exchange_fmaps), 
    #                                communication_network_def=CNNCommunicatorDilated)

    model = MultiGPU_UNet_with_comm(n_channels=3, n_classes=1, input_shape=(2560, 2560), num_comm_fmaps=num_comm_fmaps, devices=devices, depth=depth,
                                   subdom_dist=subdomain_dist, bilinear=False, comm=comm, complexity=complexity, dropout_rate=0.0, kernel_size=kernel_size, 
                                   padding=padding, communicator_type=None, comm_network_but_no_communication=(not exchange_fmaps), 
                                   communication_network_def=CNNCommunicatorDilated)
    
    
    input_shape = (3, domain_size // subdomains_dist[0], domain_size // subdomains_dist[1])  # Example input shape (channels, height, width)
    target_layer_idx = 0  # The index of the target layer in the model (e.g., the first layer in the encoder)
    
    plot_receptive_field(model, input_shape)
