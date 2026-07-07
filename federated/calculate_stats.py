import torch
import torchvision
import torchvision.transforms as transforms

def main():
    print("Loading Fashion-MNIST dataset...")
    # Load dataset - similar to notebook
    trainset = torchvision.datasets.FashionMNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transforms.ToTensor()
    )

    print(f"Dataset loaded. converting to float tensor for calculation...")
    # trainset.data is (N, H, W) uint8. Convert to float and normalize to [0, 1]
    data = trainset.data.float() / 255.0

    print("Calculating statistics...")
    mean = data.mean().item()
    std = data.std().item()

    print(f"\nFashion-MNIST Statistics:")
    print(f"Mean: {mean:.4f}")
    print(f"Std Dev: {std:.4f}")

if __name__ == "__main__":
    main()
