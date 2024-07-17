import argparse
import glob
import json
from os.path import join

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

from monai.metrics import DiceMetric, SurfaceDistanceMetric, HausdorffDistanceMetric
from monai.utils import set_determinism
from monai.networks.nets import UNet, SegResNet
from monai.networks.layers.factories import Norm

import torch

from networks import MdResUNet
from utils import sliding_window_inference


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

    with open(join(args.pdir, cf.export_params.save_dir, args.job, 'best_outputs', f'f{args.fold}', 'inferencemetrics.json'),
              'w') as outfile:
        json.dump(testmetrics, outfile, indent=4)

    return


def main(args, cf):

    train_images = sorted(
        glob.glob(join(args.pdir, cf.import_params.target_dir, "targets/*.nii.gz"))
    )
    train_labels = sorted(
        glob.glob(join(args.pdir, cf.import_params.target_dir, "masks/*.nii.gz"))
    )
    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(train_images, train_labels)
    ]

    test_files = data_dicts

    # ---> do augmentation
    _, val_trans = preprocessing(cf)

    # ---> get datasets
    test_ds = Dataset(test_files, transform=val_trans)

    test_loader = DataLoader(test_ds, batch_size=1, num_workers=0)

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

    checkpoint = torch.load(join(args.pdir, cf.export_params.save_dir, args.job, 'best_outputs', f'f{args.fold}',
                                     "checkpoint.pth.tar"), map_location=torch.device('cpu'))
    print("=> Loading Checkpoint")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        test(test_loader, model, device, cf, args)


if __name__ == '__main__':

    def str2bool(v):
        """
            Workaround to pass boolean to argparse
        """
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description='neural network training')
    parser.add_argument('--pdir', type=str, default=".", help='project directory')
    parser.add_argument('--job', type=str, default='baseline', help="")
    parser.add_argument('--fold', type=int, default=1, help='specify the fold to process')
    parser.add_argument('--inference', type=str2bool, default=True)
    args = parser.parse_args()

    config_loc = join(args.pdir, 'config', args.job, f'f{args.fold}.json')
    cf = load_conf_file(config_loc)

    set_determinism(seed=0)
    main(args, cf)
