import os.path as op

import torch
from torch.utils.data.distributed import DistributedSampler

from domain_utils.domain_loader import DomainLoader
from domain_utils.domain_bus import DomainBus

from voc_dataset import VOCDataSet
from train_utils import create_aspect_ratio_groups, GroupedBatchSampler



def init_domain(hparams, data_transform, only_val=False):

    dirs = hparams.domain_info["data_dir"]

    domainloaders = []
    val_dataloaders = []
    train_val_dataloaders = []
    train_samplers = []

    if only_val:
        d = hparams.domain_info["data_dir"][hparams.tdi]
        domain_name = op.basename(d)
        print(f"Loading Domain: {domain_name} from {d} ...")

        val_dataset = VOCDataSet(
            data_dir=d,
            transforms=data_transform["val"],
            max_samples=-1,
            num_classes=hparams.num_classes,
            split_dir='ImageSets/',
            split='val',
            dataset_name=domain_name,
            class_file=hparams.cls
        )

        print(f"Valid size: {len(val_dataset)}")

        val_data_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=hparams.batch_size, pin_memory=True,
            num_workers=hparams.workers, collate_fn=val_dataset.collate_fn)

        return val_data_loader



    print("Loading Domains ...... ")
    for seq, di in enumerate(hparams.sdi):

        d = dirs[di]
        domain_name = op.basename(d)

        print(f"Loading Domain: {domain_name} from {d} ...")

        train_dataset = VOCDataSet(
            data_dir=d,
            transforms=data_transform["train"],
            max_samples=hparams.max_samples,
            num_classes=hparams.num_classes,
            split_dir='ImageSets/',
            split='train',
            dataset_name=domain_name,
            class_file=hparams.cls,
            d_seq=seq
        )

        print(f"Train size: {len(train_dataset)}")

        val_dataset = VOCDataSet(
            data_dir=d,
            transforms=data_transform["val"],
            num_classes=hparams.num_classes,
            split_dir='ImageSets/',
            max_samples=-1,
            split='val',
            dataset_name=op.basename(d) + '_val',
            class_file=hparams.cls
        )

        print(f"Valid size: {len(val_dataset)}")
        train_val_dataset = None
        if hparams.valid_train_set:

            train_val_dataset = VOCDataSet(
                data_dir=d,
                transforms=data_transform['train'],
                num_classes=hparams.num_classes,
                split_dir='ImageSets/',
                max_samples=1000,
                dataset_name=op.basename(d) + '_trainval',
                d_seq=seq
            )

            print(f"Valid(train) size: {len(train_val_dataset)}")

        train_val_sampler = None

        if hparams.distributed:
            train_sampler = DistributedSampler(train_dataset)
            val_sampler = DistributedSampler(val_dataset)
            if hparams.valid_train_set:
                train_val_sampler = DistributedSampler(train_val_dataset)
        else:
            train_sampler = torch.utils.data.RandomSampler(train_dataset)
            val_sampler = torch.utils.data.SequentialSampler(val_dataset)
            if hparams.valid_train_set:
                train_val_sampler = torch.utils.data.SequentialSampler(train_val_dataset)

        if hparams.aspect_ratio_group_factor >= 0:
            # 统计所有图像比例在bins区间中的位置索引
            group_ids = create_aspect_ratio_groups(train_dataset, k=hparams.aspect_ratio_group_factor)
            train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, hparams.batch_size)
        else:
            train_batch_sampler = torch.utils.data.BatchSampler(
                train_sampler, hparams.batch_size, drop_last=True)

        train_dataloader = DomainLoader(
            train_dataset,
            domain_name=train_dataset.dataset_name,
            batch_sampler=train_batch_sampler,
            num_workers=hparams.workers,
            collate_fn=train_dataset.collate_fn)

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=hparams.batch_size, pin_memory=True,
            sampler=val_sampler, num_workers=hparams.workers,
            collate_fn=val_dataset.collate_fn)

        train_val_dataloader = None

        if hparams.valid_train_set:

            train_val_dataloader = torch.utils.data.DataLoader(
                train_val_dataset, batch_size=hparams.batch_size, pin_memory=True,
                sampler=train_val_sampler, num_workers=hparams.workers,
                collate_fn=train_val_dataset.collate_fn)

        domainloaders.append(train_dataloader)
        val_dataloaders.append(val_dataloader)

        if hparams.valid_train_set:
            train_val_dataloaders.append(train_val_dataloader)

        train_samplers.append(train_sampler)

    domainbus = DomainBus(domainloaders=domainloaders, iter_num=hparams.iter_num, train_samplers=train_samplers)



    return domainbus, val_dataloaders, train_val_dataloaders



