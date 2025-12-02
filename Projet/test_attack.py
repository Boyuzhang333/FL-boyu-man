#!/usr/bin/env python3
"""
Test script: Verify label flipping attack functionality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from tqdm import tqdm
import prepare_dataset

# Device configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class Net(nn.Module):
    """Simple CNN model"""
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def test_label_flipping_attack():
    """Test label flipping attack"""
    print("Testing Label Flipping Attack...")
    print(f" Device: {DEVICE}")
    
    # Create model and data
    net = Net().to(DEVICE)
    trainloader, _, _ = prepare_dataset.get_data_loader(5, 0, data_split='iid')
    
    # Get a batch of data for testing
    inputs, original_labels = next(iter(trainloader))
    inputs = inputs.to(DEVICE)
    original_labels = original_labels.to(DEVICE)
    
    print(f" Batch size: {len(original_labels)}")
    print(f"  Original labels (first 10): {original_labels[:10].cpu().numpy()}")
    
    # Apply label flipping attack
    flipped_labels = (original_labels + 1) % 10
    print(f"Flipped labels (first 10):  {flipped_labels[:10].cpu().numpy()}")
    
    # Verify attack effect
    different_count = (original_labels != flipped_labels).sum().item()
    print(f"Labels changed: {different_count}/{len(original_labels)} ({100*different_count/len(original_labels):.1f}%)")
    
    # Test random trigger
    print("\n Testing random attack trigger (10 rounds):")
    for round_num in range(10):
        execute_attack = random.random() < 0.5
        status = "ON" if execute_attack else "OFF"
        print(f"   Round {round_num+1}: Label flipping attack: {status}")
    
    print("\n Label flipping attack test completed successfully!")

if __name__ == "__main__":
    test_label_flipping_attack()
