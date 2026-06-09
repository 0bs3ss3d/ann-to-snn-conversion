"""
Extended ANN architectures: VGG16 and ResNet18 for comparison.

Used to compare different architectures' behavior in ANN-to-SNN conversion.
"""

import torch
import torch.nn as nn
from typing import List


class VGG16(nn.Module):
    """
    VGG16 architecture adapted for CIFAR-10/GTSRB (32x32 images).
    
    Original VGG16 designed for ImageNet (224x224).
    Adapted version:
    - Fewer FC layers
    - Smaller intermediate dimensions
    - Suitable for SNN conversion (no batch norm)
    """
    
    def __init__(self, num_classes: int = 10, input_channels: int = 3):
        """
        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels (3 for RGB)
        """
        super(VGG16, self).__init__()
        
        self.num_classes = num_classes
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32 -> 16
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16 -> 8
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8 -> 4
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4 -> 2
            
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 2 -> 1
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes, bias=True)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """
        Initialize weights with He initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, 3, 32, 32)
            
        Returns:
            Output logits (batch, num_classes)
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResNet18(nn.Module):
    """
    ResNet18 architecture adapted for CIFAR-10/GTSRB.
    
    Key features:
    - Residual connections for easy gradient flow
    - No batch normalization (for SNN conversion)
    - Suitable for small images (32x32)
    """
    
    def __init__(self, num_classes: int = 10, input_channels: int = 3):
        """
        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels
        """
        super(ResNet18, self).__init__()
        
        self.num_classes = num_classes
        
        # Initial conv layer
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global average pooling and classification
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes, bias=True)
        
        self._initialize_weights()
    
    def _make_layer(self, in_channels: int, out_channels: int,
                   num_blocks: int, stride: int = 1) -> nn.Sequential:
        """
        Create residual layer.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            num_blocks: Number of residual blocks
            stride: Stride for first block
            
        Returns:
            Sequential module with residual blocks
        """
        layers = []
        
        # First block (may have stride > 1)
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride))
        
        # Remaining blocks (stride = 1)
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self) -> None:
        """
        Initialize weights.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, 3, 32, 32)
            
        Returns:
            Output logits (batch, num_classes)
        """
        x = self.conv1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class ResidualBlock(nn.Module):
    """
    Basic residual block for ResNet.
    
    Architecture:
        Input
        → Conv 3x3
        → ReLU
        → Conv 3x3
        → + (residual connection)
        → ReLU
        → Output
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """
        Args:
            in_channels: Input channels
            out_channels: Output channels
            stride: Stride for first convolution
        """
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=True)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=True)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        
        out = out + identity
        out = self.relu2(out)
        
        return out
