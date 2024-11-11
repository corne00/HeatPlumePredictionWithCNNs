import torch
from models.ddu_net_no_padding import MultiGPU_UNet_padding
from models.sub_modules import CNNCommunicator
from hyperparam_optuna import init_data
from utils import parse_args

devices = ['cpu'] if not torch.cuda.is_available() else ['cuda:0']

"""
In this script, the working of the DDU-Net without padding is demonstrated. Also, it is demonstrated how
larger kernel sizes can be activated for this architecture. 

Explanation of some of the arguments of the function below:

- kernel size (int)             : kernel size used for convolutions in encoder and decoder.
- padding (int)                 : (zero) padding used for convolutions in encoder and decoder. If padding = 0, no padding is used (so the output will be smaller than the input). If padding = None, it is initialized to the 'same' (padding = kernel_size // 2)
- depthwise_conv (bool)         : if set to True, depthwise convolution is used to save memory and reduce parameters. If False, standard convolution is used.

The following two arguments are also added, but irrelevant for the case where the communication network is unused (comm = False),
which was the case for the hyperparameter optimization so far.

- kernel_size_conv (int)        : kernel size for the communication network. If set to None, this will be set to equal the kernel size in encoder and decoder.
- padding_conv (int)            : (zero) padding used in the communication network. If set to None, this will be equal to the padding of the encoder/decoder network

"""

ddunet  = MultiGPU_UNet_padding(n_channels=3, n_classes=1, input_shape=(512, 512), num_comm_fmaps=16, 
                                devices=devices, depth=3, subdom_dist=(1,1), bilinear=False, 
                                comm=False, complexity=16, dropout_rate=0.0, kernel_size=13, 
                                comm_network_but_no_communication=False, communication_network_def=CNNCommunicator, 
                                num_convs=2, padding=0, depthwise_conv=True, kernel_size_comm=None, padding_comm=None)

"""
To show the use of depthwise convolutions, consider the following example, where we compare the number of parameters
for standard and depthwise convolution for several kernel sizes:
"""
def count_parameters(model):
    """Returns the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

for kernel_size in [3,5,7,13,19,25,31]:
    ddunet_depthwise = MultiGPU_UNet_padding(n_channels=3, n_classes=1, input_shape = (1024,1024), num_comm_fmaps=16, devices=devices, depth=3, subdom_dist=(1,1), 
           comm=False, complexity=16, dropout_rate=0.0, kernel_size=kernel_size, comm_network_but_no_communication=False, 
           communication_network_def=CNNCommunicator, num_convs=2, padding=None, depthwise_conv=True, kernel_size_comm=None, padding_comm=None)
    
    ddunet_baseline = MultiGPU_UNet_padding(n_channels=3, n_classes=1, input_shape = (1024,1024), num_comm_fmaps=16, devices=devices, depth=3, subdom_dist=(1,1), 
           comm=False, complexity=16, dropout_rate=0.0, kernel_size=kernel_size, comm_network_but_no_communication=False, 
           communication_network_def=CNNCommunicator, num_convs=2, padding=None, depthwise_conv=False, kernel_size_comm=None, padding_comm=None)
    
    print(rf"Number of parameters for kernel size {kernel_size}: 1. baseline: {count_parameters(ddunet_baseline)} - 2. depthwise convolution: {count_parameters(ddunet_depthwise)}")

"""
OUTPUT:
Number of parameters for kernel size 3 : 1. baseline: 482737 - 2. depthwise convolution: 99452
Number of parameters for kernel size 5 : 1. baseline: 1261745 - 2. depthwise convolution: 110508
Number of parameters for kernel size 7 : 1. baseline: 2430257 - 2. depthwise convolution: 127092
Number of parameters for kernel size 13: 1. baseline: 8272817 - 2. depthwise convolution: 210012
Number of parameters for kernel size 19: 1. baseline: 17620913 - 2. depthwise convolution: 342684
Number of parameters for kernel size 25: 1. baseline: 30474545 - 2. depthwise convolution: 525108
Number of parameters for kernel size 31: 1. baseline: 46833713 - 2. depthwise convolution: 757284


Now, we show an example of a simple training loop for the case where we don't use zero padding. This means we need to adjust
the mask size according to the output shape of the network. This is done as follows
"""

# Set up DDUNet
ddunet  = MultiGPU_UNet_padding(n_channels=3, n_classes=1, input_shape=(640, 640), num_comm_fmaps=16, 
                                devices=devices, depth=4, subdom_dist=(1,1), bilinear=False, 
                                comm=False, complexity=16, dropout_rate=0.0, kernel_size=7, 
                                comm_network_but_no_communication=False, communication_network_def=None, 
                                num_convs=2, padding=0, depthwise_conv=True)

output_shape = ddunet.get_output_shape(input_shape=(640, 640))
print("Output shape of DDUNet:", output_shape)  # Print (464, 464) for this example

# # Set up datasets and dataloaders. Note that we use the crop size as argument to the init data
# args = parse_args()
# image_dir = "/scratch/e451412/data/dataset_large_square_6hp_varyK_1000dp inputs_pki outputs_t/Inputs"
# mask_dir = "/scratch/e451412/data/dataset_large_square_6hp_varyK_1000dp inputs_pki outputs_t/Labels"
# args.batch_size_training = 4

# dataloaders, datasets = init_data(args=args, image_dir=image_dir, mask_dir=mask_dir, crop_size=output_shape)

# # Optimizer and loss function
# optimizer = torch.optim.Adam(ddunet.parameters(), lr=0.001)
# criterion = torch.nn.MSELoss()

# # Training loop
# num_epochs = 20  # or however many epochs you need
# for epoch in range(num_epochs):
#     ddunet.train()  # Set model to training mode
    
#     running_loss = 0.0
#     for images, masks in dataloaders['train']:
#         optimizer.zero_grad()  # Zero gradients before each step
        
#         outputs = ddunet([im.float() for im in images])             # If we train on CPU, we need to convert the images to float first (they are loaded with half precision)
#         loss = criterion(outputs, masks.float().to(devices[0]))

#         # Print sizes to make sure they match
#         print("Outputs and masks shape", outputs.shape, masks.shape)

#         # Backward pass and optimization
#         loss.backward()  
#         optimizer.step()  

#         running_loss += loss.item() * masks.size(0)

#     epoch_loss = running_loss / len(dataloaders['train'].dataset)
#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}", flush=True)

# print("Training completed.")

