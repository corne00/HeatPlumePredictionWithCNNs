import os
import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_results(model, savepath, epoch_number, dataloaders):
    def plot_subplot(position, image, title='', vmin=None, vmax=None, colorbar=False):
        plt.subplot(4, 3, position)
        plt.axis("off")
        plt.imshow(image, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        if colorbar:
            plt.colorbar(shrink=0.5)
        if title:
            plt.title(title)

    def process_and_plot(images, masks, start_pos, title='', colorbar=True):
        model.eval()
        with torch.no_grad():
            predictions = model([img.float().unsqueeze(0) for img in images]).cpu()
            full_images = model.concatenate_tensors([img.unsqueeze(0) for img in images]).squeeze().cpu()

        for i in range(3):
            title = (title if i ==2 else '')
            plot_subplot(start_pos + i, full_images[i].cpu(), title=title)
        
        plot_subplot(start_pos + 3, predictions[0, 0].cpu(), vmin=0, vmax=1)
        plot_subplot(start_pos + 4, masks.cpu()[0])
        plot_subplot(start_pos + 5, masks.cpu()[0] - predictions[0, 0].cpu(), colorbar=colorbar)

    plt.figure(figsize=(9, 12))
    
    # Adjust spacing between plots
    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    train_image, train_mask = dataloaders["train"].dataset[0]
    process_and_plot(train_image, train_mask, 1, title='training', colorbar=True)

    val_image, val_mask = dataloaders["val"].dataset[0]
    process_and_plot(val_image, val_mask, 7, title='validation', colorbar=True)

    os.makedirs(savepath / "figures", exist_ok=True)
    plt.savefig(savepath / "figures" / f"epoch_{epoch_number}.png", bbox_inches='tight')
    plt.close()


# Compute average training loss per epoch
def average_training_losses(training_losses, dataloader_train, num_epochs):
    num_batches_per_epoch = len(dataloader_train)
    avg_training_losses = []
    
    for epoch in range(num_epochs):
        start_idx = epoch * num_batches_per_epoch
        end_idx = start_idx + num_batches_per_epoch
        avg_training_losses.append(np.mean(training_losses[start_idx:end_idx]))
    
    return avg_training_losses

# Code to plot val and training loss in log-log scale
def plot_losses(avg_training_losses, val_losses, save_path):
    plt.figure(figsize=(10, 6))
    
    # Plot training and validation losses on a log-log scale
    plt.semilogy(avg_training_losses, label='Training Loss (Per Epoch)', color='blue')
    plt.semilogy(val_losses, label='Validation Loss', color='red')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Save the plot to the save_path directory
    plt.savefig(os.path.join(save_path, 'losses_log.png'), bbox_inches='tight')
    plt.close()