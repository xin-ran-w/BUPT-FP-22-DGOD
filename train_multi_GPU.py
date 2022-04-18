import os
import time
import torch
import datetime

from algorithms.algorithms import get_algorithm_class
from train_utils import init_distributed_mode, save_on_master
from domain_utils.init_domain import init_domain
from params.basic_params import print_args


def main(hparams):
    init_distributed_mode(hparams)
    device = torch.device(hparams.device)

    algorithm_class = get_algorithm_class(hparams.algorithm)
    hparams = algorithm_class.get_hparams(hparams)
    print_args(hparams)

    image_transform = algorithm_class.get_transforms(hparams)
    print("Loading data")
    domainbus, val_dataloaders, t_v_loaders = init_domain(hparams=hparams, data_transform=image_transform)

    print(f"Creating model: {hparams.net}")
    # create model no background
    model = algorithm_class.get_model(hparams)
    model.to(device)

    model_without_ddp = model
    if hparams.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[hparams.gpu])
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=hparams.lr, momentum=hparams.momentum, weight_decay=hparams.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if hparams.amp else None

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=hparams.lr_steps, gamma=hparams.lr_gamma)

    # 如果传入resume参数，即上次训练的权重地址，则接着上次的参数训练
    if hparams.resume:
        # If map_location is missing, torch.load will first load the module to CPU
        # and then copy each parameter to where it was saved,
        # which would result in all processes on the same machine using the same set of devices.
        checkpoint = torch.load(hparams.resume, map_location='cpu')  # 读取之前保存的权重文件(包括优化器以及学习率策略)
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        hparams.start_epoch = checkpoint['epoch'] + 1
        if hparams.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

    train_loss = []
    learning_rate = []

    algorithm = algorithm_class(model, optimizer, domainbus, device, hparams, warmup=True, scaler=scaler)

    print("Start training")
    start_time = time.time()
    for epoch in range(hparams.start_epoch, hparams.epochs):
        algorithm.reset_domainbus(epoch)
        mean_loss, lr = algorithm.train_one_epoch(epoch)

        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # update learning rate
        lr_scheduler.step()

        # evaluate every epoch
        algorithm.eval_one_epoch(hparams.source_domains, val_dataloaders, t_v_loaders)

        if hparams.output_dir:
            # 只在主节点上执行保存权重操作
            if epoch % hparams.save_gap == 0 or epoch == hparams.epochs - 1:
                save_files = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'args': hparams,
                    'epoch': epoch}
                if hparams.amp:
                    save_files["scaler"] = scaler.state_dict()

                short_name = [name[:1] for name in hparams.source_domains]

                s2t = "".join(short_name).upper() + "2" + hparams.target_domain[:1].upper()

                save_on_master(save_files,
                               os.path.join(hparams.output_dir,
                                            f'{hparams.net[:6]}_{hparams.algorithm}_{s2t}_{epoch}.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    from params.train_params import TrainParams

    args = TrainParams().create()
    main(args)
