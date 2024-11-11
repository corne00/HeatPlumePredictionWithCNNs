import torch
import torch.nn as nn
import copy
from .sub_modules import Encoder, Decoder, CNNCommunicator


class MultiGPU_UNet_padding(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, input_shape, num_comm_fmaps: int, devices:list, 
                 depth: int = 3, subdom_dist: tuple = (1, 1), bilinear: bool = False, comm: bool = True, 
                 complexity:int = 32, dropout_rate: float = 0.0, kernel_size: int = 5, 
                 kernel_size_comm: bool = None, comm_network_but_no_communication: bool =False, 
                 communication_network_def=CNNCommunicator, num_convs:int = 2, padding:int = None, 
                 padding_comm: int = 0, depthwise_conv: bool = False):
        super(MultiGPU_UNet_padding, self).__init__()

        self.n_channels: int = n_channels
        self.n_classes: int = n_classes
        self.input_shape = input_shape
        self.num_comm_fmaps: int = num_comm_fmaps
        self.devices: list = devices
        self.depth: int = depth
        self.subdom_dist: tuple = subdom_dist
        self.nx, self.ny = subdom_dist
        self.bilinear: bool = bilinear
        self.comm: bool = comm
        self.complexity: int = complexity
        self.dropout_rate: float = dropout_rate
        self.kernel_size: int  = kernel_size

        # Set padding
        if padding is None:
            self.padding: int = self.kernel_size // 2
        else:
            self.padding: int = padding
        
        # Set padding and kernel size for the communication network
        if kernel_size_comm is None:
            self.kernel_size_comm = kernel_size
            self.padding_comm = padding
        else:
            self.kernel_size_comm: int = kernel_size_comm
            self.padding_comm = padding_comm
            print(rf"Initialized communication network with kernel size {kernel_size_comm} and padding {padding_comm}")
        



        self.comm_network_but_no_communication: bool = comm_network_but_no_communication
        self.communication_network_def = communication_network_def
        self.num_convs: int = num_convs
        self.depthwise_conv = depthwise_conv

        self.init_encoders()
        self.init_decoders()
        
        if self.comm:
            self.communication_network = self.communication_network_def(in_channels=num_comm_fmaps, out_channels=num_comm_fmaps,
                                                         dropout_rate=dropout_rate, kernel_size=self.kernel_size_comm, padding=self.padding_comm).to(devices[0])

    def compute_latents_fmap_size(self, input_size: int):
        """Returns the shape (in 1D) of the latent feature maps given a certain input size"""
        outp_shape = input_size
        outp_shape -= self.num_convs * ((self.kernel_size - 1) - 2 * self.padding)
        
        for _ in range(self.depth):
            outp_shape //= 2                                                            # downsampling
            outp_shape -= self.num_convs * ((self.kernel_size - 1) - 2 * self.padding)  # convolutions after downsampling
            
        corresponding_input_shape = outp_shape * 2 ** self.depth                        # compute input size corresponding to this latent feature map

        return outp_shape, corresponding_input_shape
    
    def get_output_shape(self, input_shape: tuple):
        """
        Computes the output shape given a certain input shape (per subdomain).
        
        Parameters:
        - input_shape (tuple): A tuple specifying the height and width of the input (e.g., (height, width)).
        
        Returns:
        - padding (tuple): A tuple with theheight and width of the output 
        """
        input_tensor = torch.randn(1,self.n_channels, input_shape[0], input_shape[1]).to(self.devices[0])

        # Perform forward pass
        with torch.no_grad():
            output = self.forward(input_image_list=[input_tensor for _ in range(self.nx * self.ny)])
        
        return (output.shape[2], output.shape[3])

    def init_encoders(self):
        encoder = Encoder(n_channels=self.n_channels, depth=self.depth, complexity=self.complexity,
                          dropout_rate=self.dropout_rate, kernel_size=self.kernel_size, num_convs=self.num_convs, padding=self.padding,
                          depthwise_conv=self.depthwise_conv)
        self.encoders = nn.ModuleList([copy.deepcopy(encoder.to(self._select_device(i))) for i in range(self.nx * self.ny)])

    def init_decoders(self):
        decoder = Decoder(n_channels=self.n_channels, depth=self.depth, n_classes=self.n_classes,
                          complexity=self.complexity, dropout_rate=self.dropout_rate, kernel_size=self.kernel_size, 
                          num_convs=self.num_convs, padding=self.padding, depthwise_conv=self.depthwise_conv)
        self.decoders = nn.ModuleList([copy.deepcopy(decoder.to(self._select_device(i))) for i in range(self.nx * self.ny)])

    def _select_device(self, index):
        return self.devices[index % len(self.devices)]
    
    def _synchronize_all_devices(self):
        for device in self.devices:
            torch.cuda.synchronize(device=device)
        
    def _get_list_index(self, i, j):
        return i * self.ny + j
    
    def _get_grid_index(self, index):
        return index // self.ny, index % self.ny
    
    def concatenate_tensors(self, tensors):
        concatenated_tensors = []
        for i in range(self.nx):
            column_tensors = []
            for j in range(self.ny):
                index = self._get_list_index(i, j)
                column_tensors.append(tensors[index].to(self._select_device(0)))
            concatenated_row = torch.cat(column_tensors, dim=2)
            concatenated_tensors.append(concatenated_row)

        return torch.cat(concatenated_tensors, dim=3)

    def _split_concatenated_tensor(self, concatenated_tensor):
        subdomain_tensors = []
        subdomain_height = concatenated_tensor.shape[3] // self.nx
        subdomain_width = concatenated_tensor.shape[2] // self.ny

        for i in range(self.nx):
            for j in range(self.ny):
                subdomain = concatenated_tensor[:, :, j * subdomain_height: (j + 1) * subdomain_height,
                            i * subdomain_width: (i + 1) * subdomain_width]
                subdomain_tensors.append(subdomain)

        return subdomain_tensors
        
    # Function to crop a tensor to the center with the target height and width
    def center_crop(self, tensor, target_height, target_width):
        _, _, h, w = tensor.size()
        start_x = (w - target_width) // 2
        start_y = (h - target_height) // 2
        return tensor[:, :, start_y:start_y + target_height, start_x:start_x + target_width]


    def forward(self, input_image_list):
        assert len(input_image_list) == self.nx * self.ny, "Number of input images must match the device grid size (nx x ny)."
        
        # Send to correct device and pass through encoder
        input_images_on_devices = [input_image.to(self._select_device(index)) for index, input_image in enumerate(input_image_list)]
        outputs_encoders = [self.encoders[index](input_image) for index, input_image in enumerate(input_images_on_devices)]

        # Do the communication step. Replace the encoder outputs by the communication output feature maps
        inputs_decoders = [[x.clone() for x in y] for y in outputs_encoders]
        
        for id in inputs_decoders:
            a = [x.size() for x in id]
            # print("Shape of encoder outputs before communication:", a) 

        # Do communication and communication network step 
        if self.comm:
            if not self.comm_network_but_no_communication:
                communication_input = self.concatenate_tensors([output_encoder[-1][:, -self.num_comm_fmaps:, :, :].clone() for output_encoder in outputs_encoders])
                # print("Shape of communication input:", communication_input.shape)
                communication_output = self.communication_network(communication_input)
                # print("Shape of communication output:", communication_output.shape)
                communication_output_split = self._split_concatenated_tensor(communication_output)

                for idx, output_communication in enumerate(communication_output_split):
                    target_height, target_width = output_communication.shape[2], output_communication.shape[3]
                    cropped_input = self.center_crop(inputs_decoders[idx][-1], target_height, target_width)

                    cropped_input[:, -self.num_comm_fmaps:, :, :] = output_communication
                    # print("Cropped input shape:", cropped_input.shape)
                    inputs_decoders[idx][-1] = cropped_input
                    # inputs_decoders[idx][-1][:, -self.num_comm_fmaps:, :, :] = output_communication
            
            elif self.comm_network_but_no_communication:        
                communication_inputs = [output_encoder[-1][:, -self.num_comm_fmaps:, :, :].clone() for output_encoder in outputs_encoders]
                # print("Shape of communication input:", communication_inputs[0].shape)
                communication_outputs = [self.communication_network(comm_input.to(self.devices[0])) for comm_input in communication_inputs]
                # print("Shape of communication output:", communication_inputs[0].shape)

                for idx, output_communication in enumerate(communication_outputs):
                    inputs_decoders[idx][-1][:, -self.num_comm_fmaps:, :, :] = output_communication.to(self._select_device(idx))
        
        for id in inputs_decoders:
            a = [x.size() for x in id]
            # print("Shape of encoder outputs after communication:", a) 

        # Do the decoding step
        outputs_decoders = [self.decoders[index](output_encoder) for index, output_encoder in enumerate(inputs_decoders)]
        
        for id in outputs_decoders:
            a = [x.size() for x in id]
            # print("Shape of decoder outputs:", a) 

        prediction = self.concatenate_tensors(outputs_decoders)
               
        return prediction

    def save_weights(self, save_path):
        state_dict = {
            'encoder_state_dict': [self.encoders[0].state_dict()],
            'decoder_state_dict': [self.decoders[0].state_dict()]
        }
        if self.comm:
            state_dict['communication_network_state_dict'] = self.communication_network.state_dict()
        torch.save(state_dict, save_path)

    def load_weights(self, load_path, device="cuda:0"):
        checkpoint = torch.load(load_path, map_location=device)
        encoder_state = checkpoint['encoder_state_dict'][0]
        decoder_state = checkpoint['decoder_state_dict'][0]

        for encoder in self.encoders:
            encoder.load_state_dict(encoder_state)
        for decoder in self.decoders:
            decoder.load_state_dict(decoder_state)
        if self.comm and 'communication_network_state_dict' in checkpoint:
            self.communication_network.load_state_dict(checkpoint['communication_network_state_dict'])
        else:
            print("No communication network found in dataset / no comm. network found")
            
