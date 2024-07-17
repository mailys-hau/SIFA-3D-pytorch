import argparse
import glob
import json
import time
from os.path import join
import math

import numpy as np

from data import preprocessing, postprocessing
from utils import (
    load_conf_file,
    Logger,
    load_checkpoint,
    save_checkpoint,
    saveimage,
    plotloss_metrics
)
from monai.data import (
    DataLoader,
    Dataset,
    decollate_batch,
)
from monai.losses import DiceFocalLoss
from monai.optimizers import Novograd
from monai.metrics import DiceMetric, SurfaceDistanceMetric, HausdorffDistanceMetric
from monai.utils import set_determinism
from monai.networks.nets import UNet, SegResNet
from monai.networks.layers.factories import Norm

import torch

from networks import MdResUNet
from utils import sliding_window_inference


def train(model, optimizer, loader, ds, device, scaler, loss_function, step, epoch_loss, logger):

    for batch_data in loader:
        step_start = time.time()
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device)
        )
        optimizer.zero_grad()
        # set AMP for MONAI training
        with torch.cuda.amp.autocast():

            outputs = model(inputs)

            if isinstance(outputs, list):
                background = [sum([outputs[i].narrow(1, 0, 1) for i in range(len(outputs))])]
                foreground = [outputs[i].narrow(1, 1, 1) for i in range(len(outputs))]

                outputs = torch.cat(background + foreground, dim=1)

            loss = loss_function(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        epoch_len = math.ceil(len(ds) / loader.batch_size)
        logger.info(
            f"{step}/{epoch_len}, train_loss: {loss.item():.4f}"
            f" step time: {(time.time() - step_start):.4f}"
        )

    return model, epoch_loss, step, optimizer


def val(model, loader, ds, device, loss_function, epoch, step, cf, args, val_loss, logger):
    # evaluation metric
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=False)
    dice_metric_singleLabel = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=False)

    for batch_data in loader:
        step_start = time.time()
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device)
        )
        # set AMP for MONAI validation
        with torch.cuda.amp.autocast():
            roi_size = cf.transform_params.crop_size
            sw_batch_size = 4
            outputs = sliding_window_inference(inputs, roi_size, sw_batch_size, model)

            loss = loss_function(outputs, labels)

        val_loss += loss.item()
        val_len = math.ceil(len(ds) / loader.batch_size)
        logger.info(
            f"{step}/{val_len}, val_loss: {loss.item():.4f}"
            f" step time: {(time.time() - step_start):.4f}"
        )

        if (epoch + 1) % cf.training_params.save_interval == 0:

            saveimage(batch_data, outputs, device, cf, args, epoch)

        post_label, post_pred = postprocessing(cf)

        outputs = [post_pred(i) for i in decollate_batch(outputs)]
        labels = [post_label(i) for i in decollate_batch(labels)]

        # compute metric for current iteration
        dice_metric(y_pred=outputs, y=labels)

        # compute dice metric 1 single label
        outputs1label = [torch.max(i[1:], 0, keepdim=True)[0] for i in outputs]
        labels1lablel = [torch.max(i[1:], 0, keepdim=True)[0] for i in labels]

        dice_metric_singleLabel(y_pred=outputs1label, y=labels1lablel)

    metric = dice_metric.aggregate()
    dice_metric.reset()
    metric1label = dice_metric_singleLabel.aggregate()
    dice_metric_singleLabel.reset()

    return metric, metric1label, step, val_loss


def test(loader, model, device, cf, args):
    # evaluation metric
    dice_metric = DiceMetric(include_background=False, reduction="none", get_not_nans=False)
    dice_metric_singleLabel = DiceMetric(include_background=False, reduction="none", get_not_nans=False)

    surfacedistance_metric = SurfaceDistanceMetric(include_background=False,
                                                   distance_metric="euclidean",
                                                   reduction="none",
                                                   get_not_nans=False)
    surfacedistance_metric_singleLabel = SurfaceDistanceMetric(include_background=False,
                                                               distance_metric="euclidean",
                                                               reduction="none",
                                                               get_not_nans=False)

    hausdroff_metric = HausdorffDistanceMetric(include_background=False,
                                               percentile=95,
                                               distance_metric="euclidean",
                                               reduction="none",
                                               get_not_nans=False)
    hausdroff_metric_singleLabel = HausdorffDistanceMetric(include_background=False,
                                                           percentile=95,
                                                           distance_metric="euclidean",
                                                           reduction="none",
                                                           get_not_nans=False)

    for batch_data in loader:
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device)
        )
        # set AMP for MONAI validation
        with torch.cuda.amp.autocast():
            roi_size = cf.transform_params.crop_size
            sw_batch_size = 4
            outputs = sliding_window_inference(inputs, roi_size, sw_batch_size, model)

        saveimage(batch_data, outputs, device, cf, args, cf.training_params.num_epochs - 1, test=True)

        post_label, post_pred = postprocessing(cf, test=True)

        outputs = [post_pred(i) for i in decollate_batch(outputs)]
        labels = [post_label(i) for i in decollate_batch(labels)]

        # compute metric for current iteration
        dice_metric(y_pred=outputs, y=labels)
        hausdroff_metric(y_pred=outputs, y=labels, spacing=0.5)
        surfacedistance_metric(y_pred=outputs, y=labels, spacing=0.5)

        # compute dice metric 1 single label
        outputs1label = [torch.max(i[1:], 0, keepdim=True)[0] for i in outputs]
        labels1lablel = [torch.max(i[1:], 0, keepdim=True)[0] for i in labels]

        dice_metric_singleLabel(y_pred=outputs1label, y=labels1lablel)
        hausdroff_metric_singleLabel(y_pred=outputs1label, y=labels1lablel, spacing=0.5)
        surfacedistance_metric_singleLabel(y_pred=outputs1label, y=labels1lablel, spacing=0.5)

    dice = dice_metric.aggregate()
    dice_metric.reset()
    dice1label = dice_metric_singleLabel.aggregate()
    dice_metric_singleLabel.reset()

    hd = hausdroff_metric.aggregate()
    hausdroff_metric.reset()
    hd1label = hausdroff_metric_singleLabel.aggregate()
    hausdroff_metric_singleLabel.reset()

    sd = surfacedistance_metric.aggregate()
    surfacedistance_metric.reset()
    sd1label = surfacedistance_metric_singleLabel.aggregate()
    surfacedistance_metric_singleLabel.reset()

    testmetrics = {'Dice': {}, '95%HD': {}, 'ASD': {}}

    for i in range(cf.model_params.out_channels - 1):
        testmetrics['Dice'][f'label{i + 1}'] = [dice[j][i].item() for j in range(loader.__len__())]
        testmetrics['Dice'][f'label{i + 1}avg'] = np.mean([dice[j][i].item() for j in range(loader.__len__())])

        testmetrics['95%HD'][f'label{i + 1}'] = [hd[j][i].item() for j in range(loader.__len__())]
        testmetrics['95%HD'][f'label{i + 1}avg'] = np.mean([hd[j][i].item() for j in range(loader.__len__())])

        testmetrics['ASD'][f'label{i + 1}'] = [sd[j][i].item() for j in range(loader.__len__())]
        testmetrics['ASD'][f'label{i + 1}avg'] = np.mean([sd[j][i].item() for j in range(loader.__len__())])

    testmetrics['Dice'][f'label'] = [dice1label[j].item() for j in range(loader.__len__())]
    testmetrics['Dice'][f'labelavg'] = np.mean([dice1label[j].item() for j in range(loader.__len__())])

    testmetrics['95%HD'][f'label'] = [hd1label[j].item() for j in range(loader.__len__())]
    testmetrics['95%HD'][f'labelavg'] = np.mean([hd1label[j].item() for j in range(loader.__len__())])

    testmetrics['ASD'][f'label'] = [sd1label[j].item() for j in range(loader.__len__())]
    testmetrics['ASD'][f'labelavg'] = np.mean([sd1label[j].item() for j in range(loader.__len__())])

    with open(join(args.pdir, cf.export_params.save_dir, args.job, 'best_outputs', f'f{args.fold}', 'testmetrics.json'),
              'w') as outfile:
        json.dump(testmetrics, outfile, indent=4)

    return


def main(args, cf):

    train_images = sorted(
        glob.glob(join(args.pdir, cf.import_params.source_dir, "*.nii.gz"))
    )
    train_labels = sorted(
        glob.glob(join(args.pdir, cf.import_params.mask_dir, "*.nii.gz"))
    )
    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(train_images, train_labels)
    ]
    train_files = [data_dicts[i] for i in cf.dataset_params.sources.index_train]
    val_files = [data_dicts[i] for i in cf.dataset_params.sources.index_val]
    test_files = [data_dicts[i] for i in cf.dataset_params.sources.index_test]

    # ---> do augmentation
    train_trans, val_trans = preprocessing(cf)

    # ---> get datasets
    train_ds = Dataset(train_files, transform=train_trans)
    val_ds = Dataset(val_files, transform=val_trans)
    test_ds = Dataset(test_files, transform=val_trans)

    # ---> set the loss
    lossfunction = DiceFocalLoss(include_background=False, softmax=True, to_onehot_y=True,
                                 lambda_focal=cf.loss_params.lambda_focal,
                                 lambda_dice=cf.loss_params.lambda_dice)

    # ---> get loaders
    train_loader = DataLoader(train_ds, batch_size=cf.training_params.batch_size, shuffle=True,
                              num_workers=cf.training_params.num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=cf.training_params.num_workers)

    test_loader = DataLoader(test_ds, batch_size=1, num_workers=cf.training_params.num_workers)

    # ---> get loggers
    train_logger = Logger(join(args.pdir, cf.export_params.save_dir, args.job, f'f{args.fold}'),
                          'training').get_logger()
    val_logger = Logger(join(args.pdir, cf.export_params.save_dir, args.job, f'f{args.fold}'),
                        'validation').get_logger()

    # ---> get models
    device = torch.device(cf.training_params.device)

    if cf.model_params.name == "mdresunet":
        model = MdResUNet(
            in_channels=cf.model_params.in_channels,
            out_channels=cf.model_params.out_channels,
            features=cf.model_params.features,
            num_decoders=cf.model_params.out_channels - 1,
            dropout_prob=cf.model_params.dropout
        ).to(device)

    elif cf.model_params.name == "resunet":
        model = UNet(
            spatial_dims=cf.model_params.spatial_dims,
            in_channels=cf.model_params.in_channels,
            out_channels=cf.model_params.out_channels,
            channels=cf.model_params.channels,
            strides=cf.model_params.strides,
            num_res_units=cf.model_params.num_res_units,
            norm=Norm.BATCH,
            dropout=cf.model_params.dropout
        ).to(device)

    elif cf.model_params.name == "segresnet":
        model = SegResNet(
            spatial_dims=cf.model_params.spatial_dims,
            init_filters=cf.model_params.init_filters,
            blocks_down=cf.model_params.blocks_down,
            in_channels=cf.model_params.in_channels,
            out_channels=cf.model_params.out_channels,
            dropout_prob=cf.model_params.dropout
        ).to(device)

    # ---> set optimizer
    optimizer = Novograd(model.parameters(), cf.training_params.learning_rate)
    scaler = torch.cuda.amp.GradScaler()

    if cf.training_params.flag_schedule:
        if cf.scheduler_params.scheduler == 'StepLR':
            step_size = cf.scheduler_params.schedulers['StepLR']['step_size']
            decay_rate = cf.scheduler_params.schedulers['StepLR']['gamma']
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=decay_rate)

    if cf.training_params.load_model:
        checkpoint = torch.load(join(args.pdir, cf.export_params.save_dir, args.job, f'f{args.fold}', "checkpoint.pth.tar"),
                                map_location=torch.device('cpu'))
        model, optimizer = load_checkpoint(checkpoint, model, optimizer)

        if cf.training_params.flag_schedule:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_metric = checkpoint["best_metric"]
        best_metric_epoch = checkpoint["best_metric_epoch"]
        best_metrics_epochs_and_time = checkpoint["best_metrics_epochs_and_time"]
        epoch_loss_values = checkpoint["epoch_loss_values"]
        val_loss_values = checkpoint["val_loss_values"]
        metric_values = checkpoint["metric_values"]
        epoch_times = checkpoint["epoch_times"]
        total_start = checkpoint["total_start"]

    else:
        start_epoch = 0
        best_metric = -1
        best_metric_epoch = -1
        best_metrics_epochs_and_time = [[], [], []]  # best_dice, epoch, time
        epoch_loss_values = []
        val_loss_values = []
        metric_values = []
        epoch_times = []
        total_start = time.time()

    for epoch in range(start_epoch, cf.training_params.num_epochs):
        epoch_start = time.time()
        train_logger.info("-" * 10)
        train_logger.info(f"epoch {epoch + 1}/{cf.training_params.num_epochs}")
        model.train()
        epoch_loss = 0
        step = 0  # batch

        model, epoch_loss, step, optimizer = train(model, optimizer, train_loader, train_ds,
                                                   device, scaler, lossfunction, step, epoch_loss,
                                                   train_logger)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        train_logger.info(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if cf.training_params.flag_schedule and cf.scheduler_params.scheduler == 'StepLR':
            scheduler.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            val_loss = 0
            val_step = 0

            with torch.no_grad():
                metric, metric1label, val_step, val_loss = val(model, val_loader, val_ds, device, lossfunction, epoch,
                                                               val_step, cf, args, val_loss, val_logger)

                metric_values.append(metric.tolist())

                if metric.mean().item() > best_metric:  #best model according to the average dice of anterior and posterior leaflet
                    best_metric = metric.mean().item()
                    best_metric_epoch = epoch + 1
                    best_metrics_epochs_and_time[0].append(metric.tolist())
                    best_metrics_epochs_and_time[1].append(best_metric_epoch)
                    best_metrics_epochs_and_time[2].append(
                        time.time() - total_start
                    )

                    # save best model
                    checkpoint = {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": epoch_loss
                    }

                    save_checkpoint(checkpoint, join(args.pdir, cf.export_params.save_dir, args.job,
                                                     'best_outputs', f'f{args.fold}'))

                val_logger.info(
                    f"current epoch: {epoch + 1}")
                for i in range(len(metric)):
                    val_logger.info(
                        f" mean dice label{i}: {metric[i]:.4f}")
                val_logger.info(
                    f" mean dice single label: {metric1label.item():.4f}"
                    f" best mean dice: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )

                val_loss /= val_step
                val_loss_values.append(val_loss)

                # save last model
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch_loss,
                    "best_metric": best_metric,
                    "best_metric_epoch": best_metric_epoch,
                    "best_metrics_epochs_and_time": best_metrics_epochs_and_time,
                    "epoch_loss_values": epoch_loss_values,
                    "val_loss_values": val_loss_values,
                    "metric_values": metric_values,
                    "epoch_times": epoch_times,
                    "total_start": total_start,
                }
                if cf.training_params.flag_schedule:
                    checkpoint["scheduler_state_dict"] = scheduler.state_dict()
                save_checkpoint(checkpoint, join(args.pdir, cf.export_params.save_dir, args.job, f'f{args.fold}'))

        train_logger.info(
            f"time consuming of epoch {epoch + 1} is:"
            f" {(time.time() - epoch_start):.4f}"
        )
        epoch_times.append(time.time() - epoch_start)

    total_time = time.time() - total_start

    train_logger.info(
        f"train completed, best_metric: {best_metric:.4f}"
        f" at epoch: {best_metric_epoch}"
        f" total time: {total_time:.4f}"
    )

    plotloss_metrics(cf, args, epoch_loss_values, val_loss_values, metric_values)

    if args.job != "xval":
        checkpoint = torch.load(join(args.pdir, cf.export_params.save_dir, args.job, 'best_outputs', f'f{args.fold}',
                                     "checkpoint.pth.tar"), map_location=torch.device('cpu'))
        print("=> Loading Checkpoint")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        with torch.no_grad():
            test(test_loader, model, device, cf, args)

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='neural network training')
    parser.add_argument('--pdir', type=str, default=".", help='project directory')
    parser.add_argument('--job', type=str, default='baseline', help="")
    parser.add_argument('--fold', type=int, default=1, help='specify the fold to process')
    args = parser.parse_args()

    config_loc = join(args.pdir, 'config', args.job, f'f{args.fold}.json')
    cf = load_conf_file(config_loc)

    set_determinism(seed=0)
    main(args, cf)
