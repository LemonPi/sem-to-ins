import os.path

import cv2
import numpy as np
from PIL import Image
import torch


class CleargraspSyntheticDataset(torch.utils.data.Dataset):
    IMG = 0
    SEM_MASK = 1
    INS_MASK = 2

    def __init__(self, root, transforms=None, allowed_class_dirs=None):
        self.root = root
        self.transforms = transforms
        self.class_dirs = ['cup-with-waves-val', 'flower-bath-bomb-val', 'heart-bath-bomb-val',
                           'square-plastic-bottle-val', 'stemless-plastic-champagne-glass-val']
        if allowed_class_dirs:
            self.class_dirs = allowed_class_dirs

        self.img_type_names = ["rgb-imgs", "segmentation-masks", "variant-masks"]
        self.img_type_paths = [[], [], []]
        self.classes = []

        # load all image files, sorting them to
        # ensure that they are aligned
        for class_name in self.class_dirs:
            imgs, sem, ins = [list(sorted(os.listdir(os.path.join(root, class_name, img_type)))) for img_type in
                              self.img_type_names]
            assert len(imgs) == len(sem)
            self.img_type_paths[self.IMG] += imgs
            self.img_type_paths[self.SEM_MASK] += sem
            self.img_type_paths[self.INS_MASK] += ins
            self.classes += [class_name] * len(imgs)

    def class_names(self):
        return ['background'] + self.class_dirs

    def get_image_full_path(self, idx, image_type=IMG):
        return os.path.join(self.root, self.classes[idx], self.img_type_names[image_type],
                            self.img_type_paths[image_type][idx])

    def __getitem__(self, idx):
        # load images ad masks
        this_class = self.classes[idx]
        # only 1 class of objects per image (and the object class doesn't seem to be encoded in the instance seg)
        label = self.class_dirs.index(this_class) + 1
        img_path = self.get_image_full_path(idx, self.IMG)
        seg_mask_path = self.get_image_full_path(idx, self.SEM_MASK)
        ins_mask_path = self.get_image_full_path(idx, self.INS_MASK)
        img = Image.open(img_path).convert("RGB")
        # TODO train on semantic segmentation instead of RGB
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        # ins_mask = Image.open(ins_mask_path)
        ins_mask = cv2.imread(ins_mask_path, cv2.IMREAD_UNCHANGED)
        ins_mask = np.uint8(ins_mask[:, :, 0])

        # instances are encoded as different colors
        obj_ids = np.unique(ins_mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = ins_mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class in the image (all of label type)
        labels = torch.ones((num_objs,), dtype=torch.int64) * label
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.img_type_paths[self.IMG])
