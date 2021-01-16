import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

from   scipy.interpolate import interp1d
from   PIL import Image
from   pyts.image import GramianAngularField


length = 448
overlap = 224
img_size = 112
max_length = 2400


def interpolate_freq(y, cur_freq, tar_freq):
    x = list(range(len(y)))
    f = interp1d(x, y, kind='cubic')
    x_pred = np.linspace(0, x[-1], num=tar_freq // cur_freq * len(y))
    y_pred = f(x_pred)
    return y_pred


def wave_preprocessing(freq, input):
    input = [float(i) for i in input.split(',')]
    input = interpolate_freq(input, freq, tar_freq=2000000).tolist()
    input = input[:max_length]
    if len(input) < max_length:
        input += [0 for _ in range(max_length-len(input))]
    return input


def slide_window(input:np.array, transformer=None):
    if transformer is None:
        transformer = GramianAngularField(image_size=112, method='summation')

    wins = []
    for i in range(len(input) // overlap):
        w = input[i * overlap : i * overlap + length]
        if len(wins) != length:
            continue
        wins.append(w)

    wins = transformer.fit_transform(wins)
    plt.figure(figsize=(8, 8))
    plt.axis('off')

    imgs = []
    for w in wins:
        img = plt.imshow(w)
        img = img.make_image(renderer=None)[0]
        imgs.append(Image.fromarray(img[::-1, :, :3]))
        plt.close('all')

    return imgs


def imgtf(imgs:list):
    vlm = torch.zeros((3, len(imgs), img_size, img_size))
    tf = transforms.Compose([
             transforms.Resize((img_size, img_size)),
             transforms.ToTensor(),
             transforms.Normalize([0, 0, 0], [1, 1, 1])])

    for i, im in enumerate(imgs):
        vlm[:, i] = tf(im)
    return vlm


def wavetf(freq, input, tf):
    input = wave_preprocessing(freq, input)
    imgs = slide_window(input, transformer=tf)
    vlm = imgtf(imgs)  # [3, 9, 112, 112]
    return vlm


def build_tf():
    tf = GramianAngularField(image_size=img_size, method='summation')
    return tf
