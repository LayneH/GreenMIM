# Written by Lang Huang (laynehuang@outlook.com)
# All rights reserved.
# --------------------------------------------------------
import os
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

try:
    import mc
except ImportError:
    mc = None
import io


class DatasetCache(data.Dataset):
    def __init__(self):
        super().__init__()
        self.initialized = False
    

    def _init_memcached(self):
        if not self.initialized:
            server_list_config_file = "/mnt/cache/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/cache/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def load_image(self, filename):
        if mc is not None:
            self._init_memcached()
            value = mc.pyvector()
            self.mclient.Get(filename, value)
            value_str = mc.ConvertBuffer(value)
            
            buff = io.BytesIO(value_str)
            with Image.open(buff) as img:
                img = img.convert('RGB')
        else:
            with Image.open(filename) as img:
                img = img.convert('RGB')
        return img



class BaseDataset(DatasetCache):
    def __init__(self, mode='train', max_class=1000, aug=None, 
                        prefix='/mnt/cache/share/images/meta',
                        image_folder_prefix='/mnt/cache/share/images/'):
        super().__init__()
        self.initialized = False

        if mode == 'train':
            image_list = os.path.join(prefix, 'train.txt')
            self.image_folder = os.path.join(image_folder_prefix, 'train')
        elif mode == 'test':
            image_list = os.path.join(prefix, 'test.txt')
            self.image_folder = os.path.join(image_folder_prefix, 'test')
        elif mode == 'val':
            image_list = os.path.join(prefix, 'val.txt')
            self.image_folder = os.path.join(image_folder_prefix, 'val')
        else:
            raise NotImplementedError('mode: ' + mode + ' does not exist please select from [train, test, val]')


        self.samples = []
        with open(image_list) as f:
            for line in f:
                name, label = line.split()
                label = int(label)
                if label < max_class:
                    self.samples.append((label, name))

        if aug is None:
            if mode == 'train':
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]),
                ])

        else:
            self.transform = aug


def get_keep_index(samples, percent, num_classes, shuffle=False):
    labels = np.array([sample[0] for sample in samples])
    keep_indexs = []
    for i in range(num_classes):
        idx = np.where(labels == i)[0]
        num_sample = len(idx)
        label_per_class = min(max(1, round(percent * num_sample)), num_sample)
        if shuffle:
            np.random.shuffle(idx)
        keep_indexs.extend(idx[:label_per_class])

    return keep_indexs


class ImageNet(BaseDataset):
    def __init__(self, mode='train', max_class=1000, num_classes=1000, transform=None, 
                       percent=1., shuffle=False, **kwargs):
        super().__init__(mode, max_class, aug=transform, **kwargs)

        assert 0 <= percent <= 1
        if percent < 1:
            keep_indexs = get_keep_index(self.samples, percent, num_classes, shuffle)
            self.samples = [self.samples[i] for i in keep_indexs]

    def __len__(self):
        return self.samples.__len__()

    def __getitem__(self, index):
        label, name = self.samples[index]
        filename = os.path.join(self.image_folder, name)
        img = self.load_image(filename)
        return self.transform(img), label


class ImageNetWithIdx(BaseDataset):
    def __init__(self, mode='train', max_class=1000, num_classes=1000, transform=None, 
                       idx=None, shuffle=False, **kwargs):
        super().__init__(mode, max_class, aug=transform, **kwargs)

        assert idx is not None

        with open(idx, "r") as fin:
            samples = [line.strip().split(" ") for line in fin.readlines()]
        self.samples = samples
        print(f"Len of training set: {len(self.samples)}")

    def __len__(self):
        return self.samples.__len__()

    def __getitem__(self, index):
        label, name = self.samples[index]
        filename = os.path.join(self.image_folder, name)
        img = self.load_image(filename)
        return self.transform(img), int(label), index
