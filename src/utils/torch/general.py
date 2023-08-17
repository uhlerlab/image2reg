import torch
from torchvision.transforms import (
    Compose,
    ToPILImage,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    ToTensor,
)


def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device
    # return torch.device("cpu")


def get_transformation_dict_for_train_val_test():
    train_transforms = Compose(
        [
            ToPILImage(),
            # Resize(64),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            ToTensor(),
        ]
    )
    val_transforms = Compose(
        [
            ToPILImage(),
            # Resfrom src.utils.torch.general import get_deviceize(64),
            ToTensor(),
        ]
    )
    test_transforms = val_transforms
    transformation_dict = {
        "train": train_transforms,
        "val": val_transforms,
        "test": test_transforms,
    }
    return transformation_dict
