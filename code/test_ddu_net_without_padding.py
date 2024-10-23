import torch
from models.ddu_net_no_padding import MultiGPU_UNet_with_comm_overlap
from models.sub_modules import CNNCommunicator

devices = ['cpu']

unet  = MultiGPU_UNet_with_comm_overlap(n_channels=3, n_classes=1, input_shape=(512, 512), num_comm_fmaps=16, devices=devices, depth=3, subdom_dist=(1,1),
                 bilinear=False, comm=False, complexity=32, dropout_rate=0.0, kernel_size=5,  
                 comm_network_but_no_communication=False, communication_network_def=CNNCommunicator, num_convs=2, padding=0)

data = [torch.randn(1,3,512, 512) for _ in range(1)]
output = unet(data)

print(output.shape)