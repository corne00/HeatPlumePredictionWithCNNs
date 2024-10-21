import torch.nn as nn
import torch

class CNNCommunicatorDilated(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1, kernel_size=5, padding=2, dilation=3):
        super(CNNCommunicatorDilated, self).__init__()

        effective_padding = dilation * (kernel_size-1) // 2

        # First convolutional layer with dilation
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.dropout1 = nn.Dropout2d(p=dropout_rate)

        # Second convolutional layer with dilation
        self.conv2 = nn.Conv2d((in_channels + out_channels) // 2, (in_channels + out_channels) // 2, kernel_size=kernel_size, stride=1, padding=effective_padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d((in_channels + out_channels) // 2)
        self.dropout2 = nn.Dropout2d(p=dropout_rate)

        # First convolutional layer with dilation
        self.conv3 = nn.Conv2d((in_channels + out_channels) // 2, (in_channels + out_channels) // 2, kernel_size=kernel_size, stride=1, padding=padding, dilation=1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.dropout3 = nn.Dropout2d(p=dropout_rate)

        # Third convolutional layer with dilation
        self.conv4 = nn.Conv2d((in_channels + out_channels) // 2, out_channels, kernel_size=kernel_size, stride=1, padding=effective_padding, dilation=dilation, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.dropout4 = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)

        x = self.relu(self.bn4(self.conv4(x)))
        x = self.dropout4(x)

        return x


if __name__=="__main__":
    # Example usage:
    in_channels = 64
    out_channels = 64
    batch_size = 16
    input_size = (3, 64, 64)  # Example input size: 3 channels, 64x64 image

    model = CNNCommunicatorDilated(in_channels, out_channels, kernel_size=5, padding=2, dilation=1)
    input_data = torch.randn(batch_size, in_channels, input_size[1], input_size[2])
    output = model(input_data)

    print("Output shape:", output.shape)
