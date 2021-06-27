import random
import numpy as np
import pandas as pd
import torch
import cv2
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn.functional as F
import albumentations as A
from PIL import Image
from skimage.color import rgb2hed
from sklearn.utils import shuffle
import os


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) * 127.5
    return image_numpy.astype(imtype)


def img2tensor(image):
    aug = transforms.Compose([
                        transforms.ToTensor()
                ])
    return aug(image)


def shuffleDf(df):
    df = shuffle(df)
    df.reset_index(inplace=True, drop=True)
    return df


def color_transform(opt, image):
    """
    PIL image transformation
    """
    if opt.Training.input_format == 'gray':
        to_gray = transforms.Grayscale(3)
        transImage = to_gray(image)
    elif opt.Training.input_format == 'hed':
        Hed = rgb2hed(image)
        H_comp = Hed[:, :, 0]
        transImage = np.repeat(H_comp[:,:,np.newaxis],3,-1)
    elif opt.Training.input_format == 'ycbcr':

    else:
        raise NotImplementedError('Input format [%s] is not implemented' % opt.Training.input_format)
    return transImage


def base_aug(opt, image):
    aug = transforms.Compose([
                      transforms.ToPILImage(),
                      transforms.Resize((opt.Training.fineSize, opt.Training.fineSize))
              ])
    image = aug(image)

    image_transform = color_transform(opt, image)
    RGB_mask = img2tensor(image)
    image_transform = img2tensor(image_transform)

    return image_transform, RGB_mask / 127.5 - 1. # [-1, 1]


def simCLR_Aug(opt, image):
    '''
    Composition of augmentations stands out: random cropping and random color distortion (but we have gray or H input, so this may not be valid)
    '''
    aug = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((opt.Training.reshape, opt.Training.reshape)),
                        transforms.RandomCrop(opt.Training.fineSize),
                        transforms.RandomHorizontalFlip(0.5),
                        transforms.RandomVerticalFlip(0.5)
                ])
    RGB_mask = aug(image)
    image_transform, RGB_mask = base_aug(opt, RGB_mask)
    return image_transform, RGB_mask


def image_read(opt, imageRow, test=False, opposite=False):
    img_path = imageRow.iloc[0,0]
    image = cv2.imread(img_path, cv2.COLOR_BGR2RGB)

    if opt.Dataset.augment_fn == 'None':
        image_aug, rgb_aug = base_aug(opt, image)
    elif test:
        image_aug, rgb_aug = base_aug(opt, image)
        imge_label = imageRow.iloc[0,1]
        img_name = img_path.split('/')[-1]
        return image_aug, imge_label, img_name
    elif opt.Dataset.augment_fn == 'simCLR':
        image_aug, rgb_aug = simCLR_Aug(opt, image)
    elif opt.Dataset.augment_fn == 'mixup':
        assert type(opposite) == pd.Series, "Asserion Failed, opposite should be pandas series!"
        if random.random() < opt.Training.mixup_p:
            l = np.random.beta(opt.Training.alpha, opt.Training.alpha)
            l = max(l, 1 - l)
            opposite_path = opposite.iloc[0, 0]
            opposite_image = cv2.imread(opposite_path, cv2.COLOR_BGR2RGB)
            mixup_image = mixupF(image, opposite_image, l)
            image_aug, rgb_aug = base_aug(opt, mixup_image, to_pil=True)
        else:
            image_aug, rgb_aug = base_aug(opt, image, to_pil=True)

    return image_aug, rgb_aug


def mixupF(imageA, imageB, l):
    mixed_input = l * imageA + (1 - l) * imageB
    return np.uint8(mixed_input)

    return image_aug, rgb_aug


class My_Transform(object):
    '''
    My transform:
    '''
    def __init__(self):
        pass

    def __call__(self, sample):
        image, label, p_id = sample['image'], sample['label'], sample['p_id']
        aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
         ])

        augmented = aug(image=image)
        image_medium = augmented['image']

        return {'image': image_medium, 'label':label, 'p_id':p_id}

class My_Normalize(object):
    '''
    My Normalize (TRail)
    '''
    def __init__(self):
        pass

    def __call__(self, sample):
        image, label, p_id = sample['image'], sample['label'], sample['p_id']
        normal_aug = A.Normalize()
        augmented_img = normal_aug(image = image)
        image = augmented_img['image']
        return {'image': image, 'label':label, 'p_id':p_id}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # print(list(sample.keys()))
        image, label, p_id = sample['image'], sample['label'], sample['p_id']
        # print(image.shape, label[0].shape)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # print(image.shape)
        if not isinstance(label, int):
            image = image.transpose((0, 3, 1, 2))
            return {'image': torch.from_numpy(image),
                    'label':torch.from_numpy(label),
                    'p_id':torch.from_numpy(p_id)}
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label':torch.FloatTensor([label]),
                'p_id':torch.FloatTensor([p_id])}


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return Variable(images)
        return_images = []
        for image in images:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size-1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        return return_images


def DiffAugment(x, policy='', channels_first=True):
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x


def rand_brightness(x):
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x


def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
    return x


def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x


def rand_translation(x, ratio=0.2):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, ratio=0.5):
    if random.random() < 0.3:
        cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
        offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
            torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
        grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
        mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
        mask[grid_batch, grid_x, grid_y] = 0
        x = x * mask.unsqueeze(1)
    return x

def rand_rotate(x, ratio=0.5):
    k = random.randint(1,3)
    if random.random() < ratio:
        x = torch.rot90(x, k, [2,3])
    return x

AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
    'rotate': [rand_rotate],
}