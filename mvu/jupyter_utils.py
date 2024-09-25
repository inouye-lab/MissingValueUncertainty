from IPython.core.display_functions import display
from torch import Tensor
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image


def draw_image(image: Tensor, nrow: int = 5):
    if len(image.shape) == 3:
        image = image.unsqueeze(dim=0)
    assert len(image.shape) == 4, f"Image shape is {image.shape}"
    image = ((image.cpu() + 1) / 2).clamp(min=0.0, max=1.0)
    grid = make_grid(image, nrow=nrow)
    display(to_pil_image(grid))
