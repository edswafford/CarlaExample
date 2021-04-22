import os
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import LongTensor
import cv2
import matplotlib.pyplot as plt
import re
import albumentations as albu
import torch
import numpy as np
import segmentation_models_pytorch as smp

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

DATA_DIR = "data_lane_segmentation"

x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'train_label')

x_valid_dir = os.path.join(DATA_DIR, 'val')
y_valid_dir = os.path.join(DATA_DIR, 'val_label')


# helper function for data visualization
def visualize(**images):
    """Plot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


class CarlaLanesDataset(Dataset):
    """ Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = ['background', 'left_marker', 'right_marker']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        # random.shuffle(self.ids)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        get_label_name = lambda fn: re.sub(".png", "_label.png", fn)
        self.masks_fps = [os.path.join(masks_dir, get_label_name(image_id)) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, LongTensor(mask)

    def __len__(self):
        return len(self.ids)


def get_training_augmentation():
    train_transform = [
        albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=0., shift_limit=0.1, p=1, border_mode=0),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.6,
        ),
        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.6,
        ),
        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.6,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    return None


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor),
    ]
    return albu.Compose(_transform)


from segmentation_models_pytorch.utils import base
from segmentation_models_pytorch.utils.losses import DiceLoss
from segmentation_models_pytorch.utils.metrics import Accuracy

label_left = CarlaLanesDataset.CLASSES.index('left_marker')
label_right = CarlaLanesDataset.CLASSES.index('right_marker')


class MultiDiceLoss(base.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.BinaryDiceLossLeft = DiceLoss()
        self.BinaryDiceLossRight = DiceLoss()

    def forward(self, y_pr, y_gt):
        # print("shape y_pr:", y_pr.shape)
        # print("shape y_gt:", y_gt.shape)
        # ypr.shape=bs,3,512,1024, ygt.shape=bs,512,1024
        left_gt = (y_gt == label_left)
        right_gt = (y_gt == label_right)
        loss_left = self.BinaryDiceLossLeft.forward(y_pr[:, label_left, :, :], left_gt)
        loss_right = self.BinaryDiceLossRight.forward(y_pr[:, label_right, :, :], right_gt)
        return (loss_left + loss_right) * 0.5


loss_string = 'multi_dice_loss'
ENCODER = 'efficientnet-b0'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'softmax2d'
DEVICE = 'cuda'
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
metrics = []
loss = MultiDiceLoss()


def create_model_and_train():
    # create segmentation model with pretrained encoder
    model = smp.FPN(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CarlaLanesDataset.CLASSES),
        activation=ACTIVATION,
        # encoder_depth = 4
    )

    train_dataset = CarlaLanesDataset(
        x_train_dir,
        y_train_dir,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CarlaLanesDataset.CLASSES,
    )

    valid_dataset = CarlaLanesDataset(
        x_valid_dir,
        y_valid_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CarlaLanesDataset.CLASSES,
    )

    bs_train = 2
    bs_valid = 2
    train_loader = DataLoader(train_dataset, batch_size=bs_train, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=bs_valid, shuffle=False)



    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=1e-4),
        # dict(params=model.parameters(), lr=1e-3)
    ])

    # create epoch runners
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    # train model
    best_loss = 1e10

    for i in range(0, 5):

        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        # do something (save model, change lr, etc.)
        if best_loss > valid_logs[loss_string]:
            best_loss = valid_logs[loss_string]
            torch.save(model, './best_model_{}.pth'.format(loss_string))
            print('Model saved!')

        if i == 3:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')


def visualize_data():
    dataset = CarlaLanesDataset(x_train_dir, y_train_dir, classes=CarlaLanesDataset.CLASSES)

    image, mask = dataset[4]  # get some sample
    visualize(
        image=image,
        label=mask
    )

    #### Visualize resulted augmented images and masks
    augmented_dataset = CarlaLanesDataset(
        x_train_dir,
        y_train_dir,
        augmentation=get_training_augmentation(),
        classes=CarlaLanesDataset.CLASSES,
    )

    # same image with different random transforms
    for i in range(3):
        image, mask = augmented_dataset[1]
        visualize(image=image, label=mask)


def main():
    # visualize_data()

    file_path = Path('./best_model_multi_dice_loss.pth')
    if file_path.exists() and file_path.is_file():
        print('best_model_multi_dice_loss.pth exist skipping rebuild.')
    else:
        create_model_and_train()

    # load best saved checkpoint
    best_model = torch.load('./best_model_multi_dice_loss.pth')

    test_best_model = True
    if test_best_model:
        # create test dataset
        test_dataset = CarlaLanesDataset(
            x_valid_dir,
            y_valid_dir,
            augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(preprocessing_fn),
            classes=CarlaLanesDataset.CLASSES,
        )

        test_dataloader = DataLoader(test_dataset)

        # evaluate model on test set
        test_epoch = smp.utils.train.ValidEpoch(
            model=best_model,
            loss=loss,
            metrics=metrics,
            device=DEVICE,
        )
        logs = test_epoch.run(test_dataloader)

    # test dataset without transformations for image visualization
    test_dataset_vis = CarlaLanesDataset(
        x_valid_dir, y_valid_dir,
        classes=CarlaLanesDataset.CLASSES,
        preprocessing=get_preprocessing(preprocessing_fn)
    )

    for i in range(3):
        n = np.random.choice(len(test_dataset_vis))

        image_vis = test_dataset_vis[n][0].astype('uint8')
        image, gt_mask = test_dataset_vis[n]

        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask_left = best_model.predict(x_tensor)[0, 1, :, :]
        pr_mask_left = (pr_mask_left.cpu().numpy())

        pr_mask_right = best_model.predict(x_tensor)[0, 2, :, :]
        pr_mask_right = (pr_mask_right.cpu().numpy())

        visualize(
            ground_truth_mask=gt_mask,
            predicted_mask_left=pr_mask_left,
            predicted_mask_right=pr_mask_right
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled by user. Bye!")
