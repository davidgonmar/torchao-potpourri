import torch
import torch.nn as nn
import torchvision # noqa

def resnet18_cifar10_modifier(model):
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    )
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.maxpool = nn.Identity()
    return model

def _maybe_identity(model, fn):
    if fn is None:
        return model
    else:
        return fn(model)

def load_vision_model(
    model_str: str,
    pretrained_path: str | None = None,
    strict=True,
    modifier_before_load=None,
    modifier_after_load=None,
    model_args={},
) -> torch.nn.Module:
    _d = {
        "resnet18": "torchvision.models.resnet18",
        "mobilenet_v2": "torchvision.models.mobilenet_v2",
    }

    if model_str not in _d:
        raise ValueError(
            f"Model {model_str} not supported. Supported models: {_d.keys()}"
        )

    model = eval(_d[model_str])(**model_args)

    if pretrained_path is None and modifier_after_load is not None:
        raise ValueError(
            "modifier_after_load should be None if pretrained_path is None"
        )

    model = _maybe_identity(model, modifier_before_load)
    if pretrained_path:
        loaded = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        if isinstance(loaded, dict) and "model" in loaded:
            model.load_state_dict(loaded["model"], strict=strict)
        if isinstance(loaded, dict) and "_orig_mod" in next(
            iter(loaded.keys())
        ):  # torch.compile model:
            new_dict = {}
            for k, v in loaded.items():
                if k.startswith("_orig_mod."):
                    new_dict[k[len("_orig_mod.") :]] = v
                else:
                    new_dict[k] = v
            model.load_state_dict(new_dict, strict=strict)
        elif isinstance(loaded, dict):
            model.load_state_dict(loaded, strict=strict)
        elif isinstance(loaded, nn.Module):
            model.load_state_dict(loaded.state_dict(), strict=strict)
        else:
            raise ValueError("Loaded model is not a dict or nn.Module")
    model = _maybe_identity(model, modifier_after_load)
    return model


def get_cifar10_test_loader(batch_size=128):
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return testloader