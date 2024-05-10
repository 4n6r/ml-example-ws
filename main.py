import torch
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms as transforms


def main():
  device = (
      "cuda"
      if torch.cuda.is_available()
      else "mps"
      if torch.backends.mps.is_available()
      else "cpu"
  )
  print(f"Using {device} device")

  train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transforms.ToTensor(), download=True)
  test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor(), download=False)


if __name__ == "__main__":
    main()