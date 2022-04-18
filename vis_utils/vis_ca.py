import torch
from torchvision.transforms import ToPILImage


def vis_channel_mask(di_masks, ds_masks, index):

    di_masks = torch.cat(di_masks, dim=0).unsqueeze(0)    # [1, 5, 256]
    ds_masks = torch.cat(ds_masks, dim=0).unsqueeze(0)    # [1, 5, 256]

    to_pil = ToPILImage()
    di_image = to_pil(di_masks)
    ds_image = to_pil(ds_masks)

    di_image.save(f'/data/wangxinran/img/di_{index}.jpg')
    ds_image.save(f'/data/wangxinran/img/ds_{index}.jpg')
