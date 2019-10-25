import os
import random
import numpy as np
import numbers
from PIL import Image, ImageOps
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms, MNIST

# def RandomCrop(data, label, height, width):
#     data, rect = image.random_crop(data,(height, width))
#     label = image.fixed_crop(label, *rect)
#     return data, label

def RandomHorizontalFlip(data, label):
        if random.random() < 0.5:
            data = data.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        return data, label


class ISPRSDataset(gluon.data.Dataset):
    def __init__(self, is_train, transform=None):
        self.crop_size = 128
        self.transform = transform
        self.root = '/home/liujiahui/data_zoo/ISPRS/Vaihingen/crop/768'
        self.data, self.label = self._read_images(root=self.root, is_train=is_train)
        # delete the images smaller than the crop size
        # self.data, self.label = self._filter(self.data), self._filter(self.label)

        assert len(self.data) == len(self.label), 'data number and label number are not match'
        print('Read ' + str(len(self.data)) + ' examples')

    def __getitem__(self, item):
        img = self.data[item]
        mask = self.label[item]
        if self.transform is not None:
            sample = self.transform({'image': img, 'label': mask})
            img, mask = sample['image'], sample['label']
        else:
            img = nd.array(img).transpose((2, 0, 1))
            mask = nd.array(mask).reshape(shape=(1, 768, 768))
        return img, mask
        # return self.data[item].transpose((2, 0, 1)), self.label[item]

    def __len__(self):
        return len(self.data)

    def _filter(self, images):
        """only keep the images, which size larger than the crop size"""
        return [img for img in images if img.shape[0] > self.crop_size[0] and img.shape[1] > self.crop_size[1]]

    def _read_images(self, root, is_train=True):
        """

        :param root:
        :param is_train:
        :return:
        """
        txt_filename = root + '/pathlist/' + ('train.txt' if is_train else 'val.txt')
        with open(txt_filename, 'r') as f:
            images = f.read().split()
        n = len(images)
        topList, gtList = [None] * n, [None] * n
        for i, filename in enumerate(images):
            # topList[i] = image.imread('%s/top/%s.tif' % (root, filename))
            # gtList[i] = image.imread('%s/gt/%s.png' % (root, filename))
            topList[i] = Image.open('%s/top/%s.tif' % (root, filename)).convert('RGB')
            gtList[i] = Image.open('%s/gt/%s.png' % (root, filename))
            # return type is PIL.Image.array
        return topList, gtList


path = ('/home/liujiahui/data_zoo/ISPRS/Vaihingen')
class ISPRSSegmentation(gluon.data.Dataset):
    """
    ISPRS dataset
    """

    def __init__(self,
                 base_dir=path,
                 split='train',
                 transform=None,
                 use_dsm=True,
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        if split != 'test':
            self._base_dir = os.path.join(base_dir,'crop/768')  # ISPRS image dir
            # self._base_dir = base_dir
        elif split == 'test':
            self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'top')
        self._cat_dir = os.path.join(self._base_dir, 'gt')

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.transform = transform

        _splits_dir = os.path.join(self._base_dir, 'pathlist')

        self.im_ids = []
        self.images = []
        self.categories = []

        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                _image = os.path.join(self._image_dir, line + ".tif")
                _cat = os.path.join(self._cat_dir, line + ".png")

                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)

                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)


        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img,
                  'label': _target,
                  }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _make_img_gt_point_pair(self, index):
        # Read Image and Target
        # _img = np.array(Image.open(self.images[index]).convert('RGB')).astype(np.float32)
        # _target = np.array(Image.open(self.categories[index])).astype(np.float32)

        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.categories[index])

        return _img, _target

    def __str__(self):
        return 'ISPRS(split=' + str(self.split) + ')'


class VOCSegDataset(gluon.data.Dataset):
    def __init__(self, is_train, crop_size):
        self.mean = nd.array([0.485, 0.456, 0.406])
        self.std = nd.array([0.229, 0.224, 0.225])
        self.crop_size = crop_size
        self.voc_root = '/home/liujiahui/data_zoo/VOCdevkit/VOC2012'
        data, label = self._read_images(root=self.voc_root, is_train=is_train)
        data = self._filter(data)
        self.data = [self.normalize(img) for img in data]
        self.label = self._filter(label)
        self._classes = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                   'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor']

        self._colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]
        # change the color map to index map
        self.cm2lbl = np.zeros(256 ** 3)
        for i, cm in enumerate(self._colormap):
            self.cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i

        for label in self.label:
            label = self.colormap2label(label, self.cm2lbl)
        assert len(self.data) == len(self.label), 'data number and label number are not match'

        print('Read '+str(len(self.data))+' examples')

    def __getitem__(self, item):
        # data, label = RandomCrop(self.data[item], self.label[item], *self.crop_size)
        # data = data.transpose((2, 0, 1))
        return self.data[item].transpose((2, 0, 1)), self.label[item]

    def __len__(self):
        return len(self.data)

    def colormap2label(self, colormap, cm2lbl):
        colormap = colormap.astype('int32').asnumpy()
        idx = (colormap[:, :, 0] * 256 + colormap[:, :, 1] * 256 + colormap[:, :, 2])
        return nd.array(cm2lbl[idx])

    def normalize(self, data):
        return (data.astype('float32') / 255 - self.mean) / self.std

    def _filter(self, images):
        '''only keep the images, which size larger than the crop size'''
        return [img for img in images if img.shape[0] > self.crop_size[0] and img.shape[1] > self.crop_size[1]]

    def _read_images(self, root, is_train=True):
        '''

        :param root: data dir of VOC 2012 dataset
        :param train: bool var --> train.txt or val.txt
        :return:
        '''
        # txt_filename: storing all the train and val data list
        txt_filename = root + '/ImageSets/Segmentation/' + ('train.txt' if is_train else 'val.txt')
        with open(txt_filename, 'r') as f:
            images = f.read().split()
        n = len(images)
        data, label = [None] * n, [None] * n
        for i, filename in enumerate(images):
            data[i] = image.imread('%s/JPEGImages/%s.jpg' % (root, filename))
            label[i] = image.imread('%s/SegmentationClass/%s.png' % (root, filename))
        return data, label