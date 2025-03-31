import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from models import MultiGPU_UNet_with_comm  # Import your model
from PIL import Image, ImageDraw
import torch.nn.functional as F


# --- CONFIGURATION ---
DEVICE_LIST = ["cuda:0"] if torch.cuda.is_available() else ["cpu"]
IMG_SIZE = (128, 128)  # Small input size for quick testing
BATCH_SIZE = 12
EPOCHS = 200

# --- SIMPLE FUNCTION TO GENERATE THE "CONNECT TWO DOTS" DATASET ---
import numpy as np
import torch
from PIL import Image, ImageDraw


def generate_connect_dots_data(batch_size, img_size):
    X = np.zeros((batch_size, 1, img_size[0], img_size[1]), dtype=np.float32)  # Blank images
    Y = np.zeros((batch_size, img_size[0], img_size[1]), dtype=np.int64)  # Target: class labels (0 or 1)

    max_dist = img_size[0] // 4  # Maximum distance between dots

    for i in range(batch_size):
        x1, y1 = np.random.randint(0, img_size[1]), np.random.randint(0, img_size[0])
        x2 = np.clip(x1 + np.random.randint(-max_dist, max_dist), 0, img_size[1] - 1)
        y2 = np.clip(y1 + np.random.randint(-max_dist, max_dist), 0, img_size[0] - 1)

        img = Image.new("L", img_size, 0)
        draw = ImageDraw.Draw(img)

        dot_size = 3
        draw.ellipse((x1 - dot_size, y1 - dot_size, x1 + dot_size, y1 + dot_size), fill=255)
        draw.ellipse((x2 - dot_size, y2 - dot_size, x2 + dot_size, y2 + dot_size), fill=255)
        draw.line((x1, y1, x2, y2), fill=255, width=3)

        binary_mask = np.array(img) > 0  # Foreground pixels → 1, Background → 0
        Y[i] = binary_mask.astype(np.int64)  # Convert to class indices

        X[i, 0] = binary_mask.astype(np.float32)  # Keep X binary (0,1)

    Y_one_hot = F.one_hot(torch.tensor(Y), num_classes=2).permute(0, 3, 1, 2)  # Convert to (B, 2, H, W)

    return torch.tensor(X), Y_one_hot.float()


# --- MODEL SETUP ---
settings = {
    "model": {
        "kernel_size": 5,
        "padding": 2,
        "dropout_rate": 0.1,
        "UNet": {"num_channels": 1, "depth": 4, "complexity": 8, "num_convs": 2},
        "comm": {"num_comm_fmaps": 2, "comm": False}
    },
    "data": {"subdomains_dist": (1, 1)}
}

# Initialize model with 2×2 subdomains
model = MultiGPU_UNet_with_comm(settings, n_classes=2, input_shape=IMG_SIZE, devices=DEVICE_LIST)
model.to(DEVICE_LIST[0])

# --- TRAINING SETUP ---
optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()  # Binary segmentation task

# --- TRAINING LOOP ---
for epoch in range(EPOCHS):
    model.train()
    X_train_full, Y_train = generate_connect_dots_data(BATCH_SIZE, IMG_SIZE)
    X_train = model._split_concatenated_tensor(X_train_full)
    X_train, Y_train = X_train, Y_train.to(DEVICE_LIST[0])

    # Split into 2×2 subdomains
    # X_subdomains = torch.chunk(X_train, chunks=4, dim=2)
    # X_subdomains = [torch.chunk(x, chunks=2, dim=3) for x in X_subdomains]
    # X_subdomains = [item for sublist in X_subdomains for item in sublist]  # Flatten to list

    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, torch.argmax(Y_train, dim=1))
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")

    # Check if it's learning
    if (epoch + 1) % 5 == 0:
        model.eval()
        plt.subplot(1, 3, 1)
        plt.imshow(X_train_full[0, 0].cpu().numpy(), cmap="gray")
        plt.title("Input")
        
        plt.subplot(1, 3, 2)
        plt.imshow(Y_train[0, 0].cpu().numpy(), cmap="gray")
        plt.title("Ground Truth")

        plt.subplot(1, 3, 3)
        plt.imshow(torch.sigmoid(output[0, 0]).detach().cpu().numpy(), cmap="gray")
        plt.title("Prediction")

        plt.savefig("./figures/test.png")
        plt.show()

