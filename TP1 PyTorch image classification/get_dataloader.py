from torchvision import datasets
from torchvision.transforms import ToTensor
import torch


def get_dataloader(name_experiment, batch_size):
    if name_experiment == "mnist":
        donnee_train = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
        )

        donnee_test = datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor(),
        )
    if name_experiment == "fmnist":
        raise NotImplementedError("Question 9 !")
    else:
        raise NotImplementedError(
            "Pas encore implémenté"
        )
    print(len(donnee_train))
    print(len(donnee_test))
    donnee_loader_train = torch.utils.data.DataLoader(donnee_train, batch_size=batch_size)
    donnee_loader_test = torch.utils.data.DataLoader(donnee_test, batch_size=batch_size)

    return donnee_loader_train, donnee_loader_test