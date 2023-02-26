r""" FSS-1000 few-shot semantic segmentation dataset """
import os
import glob

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np


class DatasetLIDC(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize, annor=None):
        self.split = split
        self.benchmark = 'lidc'
        self.shot = shot
        self.annor = annor
        self.base_path = os.path.join(datapath, 'LIDC/nodule-slices-f')

        # Given predefined test split, load randomly generated training/val splits:
        # (reference regarding trn/val/test splits: https://github.com/HKUSTCV/FSS-1000/issues/7))
        # with open('train.pkl', 'rb') as f:
        #     train_set = pickle.load(f)
        # with open('test.pkl', 'rb') as f:
        #     test_set = pickle.load(f)
        with open('./data/splits/lidc/%s.txt' % split, 'r') as f:
            self.categories = f.read().split('\n')[:-1]
        self.categories = self.categories # sorted(self.categories)

        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()

        self.transform = transform

    def __len__(self):
        return len(self.img_metadata)

    def __getitem__(self, idx):
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_mask, support_imgs, support_masks, query_mask_a = self.load_frame(query_name, support_names)

        query_img = self.transform(query_img)
        query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()
        query_mask_a = torch.stack([
            F.interpolate(i.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze() for i in
            query_mask_a])

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])

        support_masks_tmp = []
        for smask in support_masks:
            smask = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
            support_masks_tmp.append(smask)
        support_masks = torch.stack(support_masks_tmp)

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,

                 'query_mask_a': query_mask_a, # masks from different annotators

                 'class_id': torch.tensor(class_sample)}

        return batch

    def load_frame(self, query_name, support_names, annotr=None):
        query_img = Image.open(query_name).convert('RGB')
        support_imgs = [Image.open(name).convert('RGB') for name in support_names]

        if self.annor is None: # randomly select a annotator
            annotr = np.random.choice(range(4), 1, replace=False)[0]
        query_id = query_name.split('/')[-1].split('.')[0]
        query_name = os.path.join(os.path.dirname(query_name)[:-6], 'mask-' + str(annotr), query_id) + '.png'
        query_name_a = [os.path.join(os.path.dirname(query_name)[:-6], 'mask-' + str(i), query_id) + '.png' for i in range(4)]
        support_ids = [name.split('/')[-1].split('.')[0] for name in support_names]
        support_names = [os.path.join(os.path.dirname(name)[:-6], 'mask-' + str(annotr), sid) + '.png' for name, sid in zip(support_names, support_ids)]

        query_mask = self.read_mask(query_name)
        query_mask_a = [self.read_mask(i) for i in query_name_a]
        support_masks = [self.read_mask(name) for name in support_names]

        return query_img, query_mask, support_imgs, support_masks, query_mask_a

    def read_mask(self, img_name):
        mask = torch.tensor(np.array(Image.open(img_name).convert('L')))
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask

    def sample_episode(self, idx):
        query_name = self.img_metadata[idx]
        query_name_ = query_name.replace("\\", "/") #### for windows
        class_sample = self.categories.index(query_name_.split('/')[-3])
        if self.split == 'val':
            class_sample += 2030
        elif self.split == 'test':
            class_sample += 2330

        support_names = []
        while True:  # keep sampling support set if query == support
            total_ins = query_name_.split('-')[-2].split('/')[-1]
            support_name = np.random.choice(range(int(total_ins)), 1, replace=False)[0] #int(query_name_.split('-')[-1].split('.')[0])
            support_name = os.path.join(os.path.dirname(query_name), total_ins + '-' + str(support_name)) + '.png'
            if query_name != support_name or (query_name == support_name and int(total_ins)==1): support_names.append(support_name)
            # if int(total_ins) == 1:
            #     print()
            # if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

        ##### for windows
        query_name = query_name.replace("\\", "/")
        support_names = [i.replace("\\", "/") for i in support_names]

        return query_name, support_names, class_sample

    def build_class_ids(self):
        if self.split == 'trn':
            class_ids = range(0, 2030)
        elif self.split == 'val':
            class_ids = range(2030, 2330)
        elif self.split == 'test':
            class_ids = range(2330, 2630)
        return class_ids

    def build_img_metadata(self):
        img_metadata = []
        for cat in self.categories:
            img_paths = sorted([path for path in glob.glob('%s/*' % os.path.join(self.base_path, cat, 'images'))])
            for img_path in img_paths:
                if os.path.basename(img_path).split('.')[1] == 'png':
                    img_metadata.append(img_path)
        print('Total (%s) images are : %d' % (self.split, len(img_metadata)))
        return img_metadata
