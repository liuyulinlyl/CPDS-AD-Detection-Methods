import torch
import torch.nn as nn
import torch.nn.functional as F

class TCNBlock(nn.Module):
    """Basic TCN residual block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.2):
        super(TCNBlock, self).__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                            padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                            padding=padding, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        # Match channel dimensions with a 1x1 convolution when needed.
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
        # Initialize weights.
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight)
    
    def forward(self, x):
        residual = x
        
        # First convolution layer.
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Second convolution layer.
        out = self.conv2(out)
        
        # Match residual dimensions.
        if self.downsample is not None:
            residual = self.downsample(residual)
        
        # Trim padded positions to match the input length.
        if out.size(-1) != residual.size(-1):
            out = out[:, :, :residual.size(-1)]
        
        # Residual connection.
        out += residual
        out = self.relu(out)
        
        return out

class TCN(nn.Module):
    """Plug-and-play TCN module."""
    def __init__(self, input_size, num_channels, kernel_size=3, dropout=0.2):
        """
        Args:
            input_size: Input feature dimension.
            num_channels: Hidden channel sizes.
            kernel_size: Convolution kernel size.
            dropout: Dropout rate.
        """
        super(TCN, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(TCNBlock(in_channels, out_channels, kernel_size, 
                                dilation, dropout))
        
        self.network = nn.Sequential(*layers)
        self.final_conv = nn.Conv1d(num_channels[-1], input_size, 1)
        
    def forward(self, x):
        """
        Input shape: (batch_size, window_size, features)
        Output shape: (batch_size, window_size, features)
        """
        # Convert to Conv1d format: (batch, channels, length).
        x = x.transpose(1, 2)
        
        # Run through the TCN network.
        out = self.network(x)
        
        # Restore channels to the input feature dimension.
        out = self.final_conv(out)
        
        # Convert back to the original layout.
        out = out.transpose(1, 2)
        
        return out

# Example usage
if __name__ == "__main__":
    # Create the TCN module.
    batch_size = 32
    window_size = 100
    features = 64
    
    # Channel settings can adjust model depth.
    num_channels = [64, 64, 64, 64]  # 4-layer TCN
    
    tcn = TCN(features, num_channels)
    
    # Create test input.
    x = torch.randn(batch_size, window_size, features)
    
    # Forward pass.
    output = tcn(x)

