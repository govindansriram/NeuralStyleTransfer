from PIL import Image
import torchvision.transforms as transforms
import torch
from matplotlib import pyplot as plt


def image_loader(desired_image_size,
                 image_path=None,
                 picture=None,
                 device='cpu'):

    if image_path is not None:
        picture = Image.open(image_path)

    # picture.show()

    pic_size = picture.size
    max_size = max(pic_size)
    picture = picture.resize((max_size, max_size))

    loader = transforms.Compose([transforms.Resize(desired_image_size),
                                 transforms.ToTensor()])
    picture = loader(picture).unsqueeze(0)
    return picture.to(device, torch.float)


def show_image(tensor, title):
    un_loader = transforms.ToPILImage()
    picture = tensor.cpu().clone()
    picture = picture.squeeze(0)
    picture = un_loader(picture)
    plt.imshow(picture)
    plt.title(title)
    plt.pause(0.1)
    plt.show()
