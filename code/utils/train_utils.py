import os

import torch
import contextlib
import time
from tqdm import tqdm
from .visualization import plot_results

def compute_validation_loss(model, loss_fn, dataloader, device, data_type, half_precision):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        with (torch.autocast(device_type='cuda', dtype=data_type) if half_precision else contextlib.nullcontext()):
            for images, masks in tqdm(dataloader, disable=True):
            for images, masks in tqdm(dataloader, disable=True):
                images = [im.half() for im in images]
                masks = masks.to(device, dtype=torch.half)

                predictions = model(images)
                loss = loss_fn(predictions, masks)
                total_loss += loss.item()
                num_batches += 1
        
    average_loss = total_loss / num_batches
    return average_loss

class CombiLoss(torch.nn.Module):
    def __init__(self, alpha:float=1):
        super(CombiLoss, self).__init__()
        self.mse = torch.nn.MSELoss()
        self.mae = torch.nn.L1Loss()
        self.alpha = alpha
    def forward(self, x, y):
        return self.alpha * self.mse(x, y) + (1-self.alpha) * self.mae(x, y)

# Train function
def train_parallel_model(model, dataloader_train, dataloader_val, train_dataset, val_dataset, scaler, data_type, half_precision, comm, num_epochs, 
                         num_comm_fmaps, save_path, subdomains_dist, exchange_fmaps, devices, num_convs,
                         padding, depth, kernel_size, complexity, communication_network=None, dropout_rate=0.0, weight_decay_adam:float=1e-5, loss_fn_alpha:float=1., lr:float=1e-4,
                         loss_func=None, val_loss_func=None):
    
    # Check to make sure  
    if num_comm_fmaps == 0:
        comm = False
    
    # Initialize the network architecture
    unet = model(n_channels=3, n_classes=1, input_shape=(640, 640), num_comm_fmaps=num_comm_fmaps, devices=devices, depth=depth,
                                   subdom_dist=subdomains_dist, bilinear=False, comm=comm, complexity=complexity, dropout_rate=dropout_rate, 
                                   kernel_size=kernel_size, padding=padding, communicator_type=None, comm_network_but_no_communication=(not exchange_fmaps), 
                                   communication_network_def=communication_network, num_convs=num_convs)
    
    unet.save_weights(save_path=os.path.join(save_path, "unet.pth"))
              
    if comm:
        parameters = list(unet.encoders[0].parameters()) + list(unet.decoders[0].parameters()) + list(unet.communication_network.parameters()) 
    else:
        parameters = list(unet.encoders[0].parameters()) + list(unet.decoders[0].parameters())
        
    optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay_adam)

    if loss_func is None:
        loss = CombiLoss(loss_fn_alpha)
    else:
        loss = loss_func
    losses = []

    # Wrap your training loop with tqdm
    start_time = time.time()
    validation_losses = []
    best_val_loss = float('inf')

    # Iterate over the epochs
    for epoch in range(num_epochs):
        unet.train()
        epoch_losses = []  # Initialize losses for the epoch
        
        for images, masks in tqdm(dataloader_train, disable=True):
            optimizer.zero_grad()
            
            with (torch.autocast(device_type='cuda', dtype=data_type) if half_precision else contextlib.nullcontext()):
            
                # Data loading and sending to the correct device
                images = ([im.half() for im in images] if half_precision else [im.float() for im in images])
                masks = masks.to(devices[0], dtype=data_type)

                ## Forward propagation:
                predictions = unet(images)

                ## Backward propagation
                l = loss(predictions, masks)
                
            scaler.scale(l).backward()

            losses.append(l.item())  # Append loss to global losses list
            epoch_losses.append(l.item())  # Append loss to epoch losses list

            # Weight upgrade of the encoders
            with torch.no_grad():
                for i in range(1, len(unet.encoders)):
                    for param1, param2 in zip(unet.encoders[0].parameters(), unet.encoders[i].parameters()):
                        if param1.grad is not None:
                            param1.grad += param2.grad.to(devices[0])
                            param2.grad = None

            # Weight update of the decoders
            with torch.no_grad():
                for i in range(1, len(unet.decoders)):
                    for param1, param2 in zip(unet.decoders[0].parameters(), unet.decoders[i].parameters()):
                        param1.grad += param2.grad.to(devices[0])
                        param2.grad = None
            
            # Set optimizer step
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            # Syncrhonize weights to the encoder clones
            for i in range(1, len(unet.encoders)):
                unet.encoders[i].load_state_dict(unet.encoders[0].state_dict())
                
            # Syncrhonize weights to the decoder clones
            for i in range(1, len(unet.decoders)):
                unet.decoders[i].load_state_dict(unet.decoders[0].state_dict())
    
        # Compute and print validation loss
        if val_loss_func is None:
            val_loss_func = torch.nn.MSELoss()
        else:
            val_loss_func = val_loss_func

        val_loss = compute_validation_loss(unet, val_loss_func, dataloader_val, devices[0], data_type=data_type, half_precision=half_precision)
        print(f'Validation Loss (Dice): {val_loss:.4f}, Train Loss: {losses[-1]:.4f}')
        
        validation_losses.append(val_loss)
        
        # Check for improvement and save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            unet.save_weights(save_path=os.path.join(save_path, "unet.pth"))
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        if (epoch + 1) % 10 == 0:
            plot_results(unet=unet, savepath=save_path, epoch_number=epoch, train_dataset=train_dataset, val_dataset=val_dataset)

        # if torch.cuda.is_available()
        #     # Track maximum GPU memory used
        #     max_memory_used = torch.cuda.max_memory_allocated() / 1024**3  # Convert to GB
        #     print(f"Maximum GPU Memory Used in Epoch {epoch+1}: {max_memory_used:.2f} GB")
        #     torch.cuda.reset_peak_memory_stats()

    print(f"Training the model {'with' if comm else 'without'} communication network took: {time.time() - start_time:.2f} seconds.")
    
    # Load the best weights
    unet.load_weights(load_path=os.path.join(save_path, "unet.pth"), device=devices[0])
    
    return unet, validation_losses, losses
