import os

import torch
import contextlib
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from .visualization import plot_results

def compute_validation_loss(model, loss_fn, dataloader, device, data_type, half_precision, verbose=False):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        with (torch.autocast(device_type='cuda', dtype=data_type) if half_precision else contextlib.nullcontext()):
            for images, masks in tqdm(dataloader, disable=(not verbose)):

                masks = masks.to(device)
                predictions = model(images)
                loss = loss_fn(predictions, masks)
                total_loss += loss
                num_batches += 1
        
    average_loss = total_loss.item() / num_batches
    return average_loss

# Train function
def train_parallel_model(model, dataloader_train, dataloader_val, train_dataset, val_dataset, scaler, data_type, half_precision, comm, num_epochs, 
                         num_comm_fmaps, save_path, subdomains_dist, exchange_fmaps, devices, num_convs,
                         padding, depth, kernel_size, complexity, communication_network=None, dropout_rate=0.0, weight_decay_adam:float=1e-5, lr:float=1e-4, plot_freq:int=10,
                         loss_func=None, val_loss_func=None, verbose=False, num_channels=3, track_loss_functions=None):
    
    # Check to make sure  
    if num_comm_fmaps == 0:
        comm = False

    writer = SummaryWriter(save_path)
    
    # Initialize the network architecture
    unet = model(n_channels=num_channels, n_classes=1, input_shape=(640, 640), num_comm_fmaps=num_comm_fmaps, devices=devices, depth=depth,
                                   subdom_dist=subdomains_dist, bilinear=False, comm=comm, complexity=complexity, dropout_rate=dropout_rate, 
                                   kernel_size=kernel_size, padding=padding, communicator_type=None, comm_network_but_no_communication=(not exchange_fmaps), 
                                   communication_network_def=communication_network, num_convs=num_convs).half()
    
    unet.save_weights(save_path=os.path.join(save_path, "unet.pth"))
              
    if comm:
        parameters = list(unet.encoders[0].parameters()) + list(unet.decoders[0].parameters()) + list(unet.communication_network.parameters()) 
    else:
        parameters = list(unet.encoders[0].parameters()) + list(unet.decoders[0].parameters())
        
    optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay_adam)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    summary_losses = {}

    # Wrap your training loop with tqdm
    start_time = time.time()
    validation_losses = []
    best_val_loss = float('inf')

    # Iterate over the epochs
    epochs = tqdm(range(num_epochs), desc="epochs", disable=False)
    for epoch in epochs:
        unet.train()
        epoch_losses = 0.0  # Initialize losses for the epoch
        
        for images, masks in tqdm(dataloader_train, disable=(not verbose)):
            optimizer.zero_grad()
            
            with (torch.autocast(device_type='cuda', dtype=data_type) if half_precision else contextlib.nullcontext()):
                
                masks = masks.to(devices[0])

                ## Forward propagation:
                predictions = unet(images)

                ## Backward propagation
                l = loss_func(predictions, masks)
                
            scaler.scale(l).backward()

            epoch_losses += l  # Add loss to epoch losses

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

        val_loss = compute_validation_loss(unet, val_loss_func, dataloader_val, devices[0], data_type=data_type, half_precision=half_precision, verbose=False)
        validation_losses.append(val_loss)
        epoch_losses = epoch_losses.item()
        # print(f'Validation Loss: {val_loss:.4f}, Train Loss: {epoch_losses:.4f}')
        epochs.set_postfix_str(f"train loss: {(epoch_losses)/len(dataloader_train):.2e}, val loss: {val_loss:.2e}, lr: {optimizer.param_groups[0]['lr']:.1e}")
        
        # Check for improvement and save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            unet.save_weights(save_path=os.path.join(save_path, "unet.pth"))

        writer.add_scalar("train_loss", epoch_losses/len(dataloader_train), epoch)
        writer.add_scalar("val_loss", val_loss, epoch)
        writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)
        for name, loss_fct in track_loss_functions.items():
            try:
                # tmp_train_loss = compute_validation_loss(unet, loss_fct, dataloader_train, devices[0], data_type=data_type, half_precision=half_precision, verbose=False)
                # writer.add_scalar(f"train-{name}", tmp_train_loss, epoch)
                # summary_losses[f"train-{name}"] = tmp_train_loss
                tmp_val_loss = compute_validation_loss(unet, loss_fct, dataloader_val, devices[0], data_type=data_type, half_precision=half_precision, verbose=False)
                writer.add_scalar(f"val-{name}", tmp_val_loss, epoch)
                summary_losses[f"val-{name}"] = tmp_val_loss
            except: pass
            
        scheduler.step(val_loss)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        if (epoch + 1) % plot_freq == 0:
            plot_results(unet=unet, savepath=save_path, epoch_number=epoch, train_dataset=train_dataset, val_dataset=val_dataset)

        # if torch.cuda.is_available()
        #     # Track maximum GPU memory used
        #     max_memory_used = torch.cuda.max_memory_allocated() / 1024**3  # Convert to GB
        #     print(f"Maximum GPU Memory Used in Epoch {epoch+1}: {max_memory_used:.2f} GB")
        #     torch.cuda.reset_peak_memory_stats()
    summary_losses["val_losses"] = validation_losses
    print(f"Training the model {'with' if comm else 'without'} communication network took: {time.time() - start_time:.2f} seconds.", flush=True)
    
    # Load the best weights
    unet.load_weights(load_path=os.path.join(save_path, "unet.pth"), device=devices[0])
    
    return unet, summary_losses
