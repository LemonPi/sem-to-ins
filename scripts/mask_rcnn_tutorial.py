import sys

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import torchvision
import torch
from PIL import Image
import cv2
import os.path
import numpy as np
from sem_to_ins import cfg
import torch.utils.data
import torchvision.transforms as TT
from sem_to_ins.dataset import CleargraspSyntheticDataset
from sem_to_ins.detection_reference import transforms as T
from sem_to_ins.detection_reference import utils
from sem_to_ins.detection_reference.engine import train_one_epoch, evaluate
import matplotlib.pyplot as plt

import random
import torch
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')
logging.getLogger('matplotlib.font_manager').disabled = True


def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def to_tensor(img):
    transform = T.Compose([T.ToTensor()])  # Defing PyTorch Transform
    img = transform(img).to(device=d)  # Apply the transform to the image
    return img


d = get_device()

p = os.path.join(cfg.DATA_DIR, 'PennFudanPed/PNGImages/FudanPed00001.png')
img = Image.open(p)
# img.show()

mask = Image.open(os.path.join(cfg.DATA_DIR, 'PennFudanPed/PedMasks/FudanPed00001_mask.png'))
mask.putpalette([
    0, 0, 0,  # black background
    255, 0, 0,  # index 1 is red
    255, 255, 0,  # index 2 is yellow
    255, 153, 0,  # index 3 is orange
])
# mask.show()

p = os.path.join(cfg.DATA_DIR, 'synthetic-val/square-plastic-bottle-val/variant-masks/000000000-variantMasks.exr')
mask2 = cv2.imread(p, cv2.IMREAD_UNCHANGED)
lowest_offset = np.min(mask2[mask2 > 0])
mask2[mask2 > 0] -= lowest_offset - 1
mask2 = np.uint8(mask2)
mim = Image.fromarray(mask2[:, :, 0])
mim.putpalette([
    0, 0, 0,  # black background
    255, 0, 0,  # index 1 is red
    255, 255, 0,  # index 2 is yellow
    255, 153, 0,  # index 3 is orange
    153, 153, 0,
    153, 153, 255,
])


# mim.show()

# mask2 = cv2.cvtColor(mask2, cv2.COLOR_BGR2RGB)  # Convert to RGB
# plt.imshow(mask2)
# plt.xticks([])
# plt.yticks([])
# plt.show()


class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

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
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
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
        return len(self.imgs)


from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    device = get_device()
    checkpoint = '/home/zhsh/catkin_ws/src/semantic_to_instance_segmentation/checkpoints/Subset.9.tar'
    train_after_loading = False
    threshold = 0.5
    test_index = 9
    torch.manual_seed(0)

    # our dataset has two classes only - background and person
    # num_classes = 2
    # dataset = PennFudanDataset(os.path.join(cfg.DATA_DIR, 'PennFudanPed/'), get_transform(train=True))
    # dataset_test = PennFudanDataset(os.path.join(cfg.DATA_DIR, 'PennFudanPed/'), get_transform(train=False))
    dataset = CleargraspSyntheticDataset(os.path.join(cfg.DATA_DIR, 'synthetic-val/'), get_transform(train=True))
    dataset_test = CleargraspSyntheticDataset(os.path.join(cfg.DATA_DIR, 'synthetic-val/'), get_transform(train=False))
    class_names = dataset.class_names()
    num_classes = len(class_names)

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    indices_train = indices[:-50]
    indices_test = indices[-50:]
    dataset = torch.utils.data.Subset(dataset, indices_train)
    dataset_test = torch.utils.data.Subset(dataset_test, indices_test)

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # load our model if it exists
    loaded_model = load(checkpoint, model, optimizer)

    if not loaded_model or train_after_loading:
        num_epochs = 10

        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            evaluate(model, data_loader_test, device=device)
            save(model, optimizer, 'cleargrasp', epoch)

    model.eval()
    instance_segmentation_api(dataset_test.dataset.get_image_full_path(indices_test[test_index]), model, threshold,
                              class_names)
    plt.show()


def get_prediction(img_path, model, threshold, class_names):
    img = Image.open(img_path)
    transform = TT.Compose([TT.ToTensor()])
    img = transform(img).to(device=get_device())

    pred = model([img])
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    pred_class = [class_names[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    masks = masks[:pred_t + 1]
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_class = pred_class[:pred_t + 1]
    return masks, pred_boxes, pred_class


def random_colour_masks(image):
    colours = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255], [80, 70, 180],
               [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190]]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0, 10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask


def instance_segmentation_api(img_path, *args, rect_th=3, text_size=1, text_th=2):
    masks, boxes, pred_cls = get_prediction(img_path, *args)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(len(masks)):
        rgb_mask = random_colour_masks(masks[i])
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=rect_th)
        cv2.putText(img, pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)
    plt.figure(figsize=(20, 30))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])


def save(model, optimizer, name, epoch):
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    base_dir = os.path.join(cfg.ROOT_DIR, 'checkpoints')
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir, exist_ok=True)
    full_name = os.path.join(base_dir, '{}.{}.tar'.format(name, epoch))
    torch.save(state, full_name)
    logger.info("saved checkpoint %s", full_name)


def load(filename, model, optimizer):
    if not os.path.isfile(filename):
        return False
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    logger.info("loaded checkpoint %s", filename)
    return True


if __name__ == "__main__":
    main()
