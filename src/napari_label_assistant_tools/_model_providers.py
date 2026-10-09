"""Lazy model providers sharing the grayscale binary-logit contract."""
from dataclasses import asdict, dataclass
from importlib.util import find_spec

_REGISTERED_PROVIDERS = {}

def register_model_provider(name, builder, *, validator=None):
    """Register trusted Python code: builder(spec, initialize=bool) -> nn.Module."""
    if name in ("custom", "standard", "smp") or not name.isidentifier() or not callable(builder):
        raise ValueError("Choose a unique provider name and callable builder.")
    _REGISTERED_PROVIDERS[name] = (builder, validator)


@dataclass(frozen=True)
class ModelSpec:
    provider: str = "custom"
    architecture: str = "unet"
    encoder: str = "resnet18"
    initialization: str = "random"
    width: int = 8
    preprocessing: str = "percentile_1_99"

    def validate(self):
        if self.provider not in ("custom", "standard", "smp") and self.provider not in _REGISTERED_PROVIDERS:
            raise ValueError(f"Model provider {self.provider!r} is unavailable. Enable or register its provider code before loading weights.")
        if self.initialization not in ("random", "imagenet"):
            raise ValueError("Unknown model initialization.")
        if self.preprocessing != "percentile_1_99":
            raise ValueError("Unsupported preprocessing.")
        if self.provider in ("custom", "standard") and (self.architecture != "unet" or self.initialization != "random"):
            raise ValueError("Built-in U-Nets use random initialization.")
        if self.provider == "custom" and self.width not in (4, 8, 12, 16):
            raise ValueError("Custom U-Net width must be 4, 8, 12 or 16.")
        if self.provider == "standard" and self.width not in (16, 32, 64):
            raise ValueError("Standard U-Net width must be 16, 32 or 64.")
        if self.provider == "smp" and self.architecture not in ("unet", "unetplusplus", "deeplabv3plus"):
            raise ValueError("Unsupported SMP architecture.")
        if self.provider in _REGISTERED_PROVIDERS:
            validator = _REGISTERED_PROVIDERS[self.provider][1]
            if validator is not None:
                validator(self)

    def label(self):
        if self.provider == "smp":
            return f"SMP {self.architecture} / {self.encoder} / {self.initialization}"
        if self.provider in ("custom", "standard"):
            return f"{self.provider.title()} U-Net / width {self.width}"
        return f"{self.provider} / {self.architecture} / width {self.width}"

    def to_dict(self):
        return asdict(self)


def spec_for(config):
    return ModelSpec(**config.model_spec) if config.model_spec else ModelSpec(width=config.base_channels)


def provider_available(provider):
    if provider == "smp":
        return find_spec("segmentation_models_pytorch") is not None
    return provider in ("custom", "standard") or provider in _REGISTERED_PROVIDERS


def build_model(config, *, initialize=True):
    spec = spec_for(config)
    spec.validate()
    from ._quick_train import _torch, build_unet
    if spec.provider in _REGISTERED_PROVIDERS:
        network = _REGISTERED_PROVIDERS[spec.provider][0](spec, initialize=initialize)
        if not isinstance(network, _torch().nn.Module):
            raise TypeError("Model provider must return a PyTorch nn.Module.")
        return network
    if spec.provider == "custom":
        return build_unet(spec.width)
    if spec.provider == "smp":
        smp = require_smp()
        constructors = {"unet": smp.Unet, "unetplusplus": smp.UnetPlusPlus, "deeplabv3plus": smp.DeepLabV3Plus}
        return constructors[spec.architecture](encoder_name=spec.encoder,
            encoder_weights="imagenet" if initialize and spec.initialization == "imagenet" else None,
            in_channels=1, classes=1, activation=None)
    torch = _torch()
    nn = torch.nn
    class Block(nn.Sequential):
        def __init__(self, a, b):
            super().__init__(nn.Conv2d(a,b,3,padding=1), nn.BatchNorm2d(b), nn.ReLU(),
                nn.Conv2d(b,b,3,padding=1), nn.BatchNorm2d(b), nn.ReLU())
    class StandardUNet(nn.Module):
        def __init__(self):
            super().__init__()
            widths = [spec.width * 2**i for i in range(4)]
            self.down = nn.ModuleList(Block(1 if i == 0 else widths[i-1], w) for i,w in enumerate(widths))
            self.transpose = nn.ModuleList(nn.ConvTranspose2d(widths[i+1], widths[i], 2, stride=2) for i in range(2,-1,-1))
            self.up = nn.ModuleList(Block(2*widths[i], widths[i]) for i in range(2,-1,-1))
            self.output = nn.Conv2d(widths[0],1,1)
        def forward(self, x):
            skips=[]
            for i,block in enumerate(self.down):
                x=block(torch.nn.functional.max_pool2d(x,2) if i else x)
                skips.append(x)
            for transpose,block,skip in zip(self.transpose,self.up,reversed(skips[:-1])):
                x=transpose(x)
                if x.shape[-2:] != skip.shape[-2:]:
                    x=torch.nn.functional.interpolate(x,size=skip.shape[-2:],mode="bilinear",align_corners=False)
                x=block(torch.cat((x,skip),dim=1))
            return self.output(x)
    return StandardUNet()


def training_mode(network, batch_size):
    """Singleton batches use running BatchNorm statistics, including pooled heads."""
    from ._quick_train import _torch
    network.train()
    if batch_size == 1:
        for module in network.modules():
            if isinstance(module, _torch().nn.modules.batchnorm._BatchNorm):
                module.eval()


def library_versions():
    from importlib.metadata import version, PackageNotFoundError
    result = {}
    for name in ("torch", "segmentation-models-pytorch", "timm", "napari-label-assistant"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            pass
    return result


def registered_provider_names():
    return sorted(_REGISTERED_PROVIDERS)


def require_smp():
    """Distinguish an absent optional package from a broken dependency import."""
    import importlib
    import sys
    import shlex
    try:
        return importlib.import_module("segmentation_models_pytorch")
    except ModuleNotFoundError as exc:
        if exc.name == "segmentation_models_pytorch":
            command = shlex.quote(sys.executable) + " -m pip install segmentation-models-pytorch"
            raise ValueError("SMP is not installed in the Python environment running napari. "
                             "Install with: " + command + ". Then restart napari.") from exc
        raise ValueError(f"SMP dependency import failed: {type(exc).__name__}: {exc}. "
                         f"Python: {sys.executable}. Check the dependency error before reinstalling SMP.") from exc
    except Exception as exc:
        raise ValueError(f"SMP is present but could not be imported: {type(exc).__name__}: {exc}. "
                         f"Python: {sys.executable}. Check compatible PyTorch, torchvision and SMP installations.") from exc
