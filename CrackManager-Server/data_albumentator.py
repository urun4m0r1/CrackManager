import albumentations as A

from config_parser import Param


def __round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)


# define heavy augmentations
def get_trn_augment():
    train_transform = [

        A.HorizontalFlip(p=0.5),

        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        A.PadIfNeeded(min_height=Param.BLOCK, min_width=Param.BLOCK, always_apply=True, border_mode=0),
        A.RandomCrop(height=Param.BLOCK, width=Param.BLOCK, always_apply=True),

        A.IAAAdditiveGaussianNoise(p=0.2),
        A.IAAPerspective(p=0.5),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightness(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.IAASharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.RandomContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        A.Lambda(mask=__round_clip_0_1)
    ]
    return A.Compose(train_transform)


# Add paddings to make image shape divisible by 32
def get_vld_augment():
    test_transform = [
        A.PadIfNeeded(Param.BLOCK, Param.BLOCK),
        A.RandomCrop(Param.BLOCK, Param.BLOCK),
    ]
    return A.Compose(test_transform)


# Construct preprocessing transform
def get_preprocess(preprocessing_fn):
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)
