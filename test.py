#Evaluate of SIFA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import yaml

from pathlib import Path
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader

from dataset import SingleDataset
from metrics import dice_eval,assd_eval,create_visual_anno
from model import SIFA
from utils import parse_config



def norm_01(image):
    mn = np.min(image)
    mx = np.max(image)
    image = (image-mn)/(mx-mn).astype(np.float32)
    return image

def save_img(image):
    image = norm_01(image)
    image = (image*255).astype(np.uint8)
    return image


def test():
    # Load config
    config = "./config/test.cfg"
    config = parse_config(config)
    exp_name = config['test']['exp_name']
    root = Path(config["test"]["save_path"]).expanduser().joinpath(exp_name)
    pmodel = root.joinpath(config["test"]["test_model"])

    device = torch.device(f"cuda:{config['test']['gpu']}")
    test_path = config['test']['test_path']
    num_classes = config['test']['num_classes']
    sifa_model = SIFA(config, "test").to(device)
    sifa_model.load_state_dict(torch.load(str(pmodel)))
    sifa_model.eval()
    #test dataset
    test_dataset = SingleDataset(test_path)
    batch_size = config['test']['batch_size']
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    #test
    all_batch_dice = []
    all_batch_assd = []
    gt_shape = (144, 144, 144)
    with torch.no_grad():
        for it,(xt, xt_label) in enumerate(test_loader):
            xt = xt.to(device)
            xt_label = xt_label.numpy().squeeze().astype(np.uint8)
            output = sifa_model.test_seg(xt).detach()
            output = output.squeeze(0)
            output = torch.argmax(output,dim=0)
            output = output.float().cpu().numpy()

            xt = xt.detach().cpu().numpy().squeeze()
            gt = xt_label.reshape(gt_shape).astype(np.uint8)
            output = output.squeeze()
            xt = save_img(xt)
            #output_vis = create_visual_anno(output)
            #gt_vis = create_visual_anno(gt)
            results = root.joinpath("evaluation-results")
            results.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(sikt.GetImageFromArray(xt), results.joinpath(f"xt-{it + 1}.nii.gz"))
            sitk.WriteImage(sikt.GetImageFromArray(gt), results.joinpath(f"gt-{it + 1}.nii.gz"))
            sitk.WriteImage(sikt.GetImageFromArray(output), results.joinpath(f"output-{it + 1}.nii.gz"))

            #FIXME: Make it a bit more pretty
            one_case_dice = dice_eval(output, xt_label, num_classes) * 100
            all_batch_dice += [one_case_dice]
            try:
                one_case_assd = assd_eval(output, xt_label, num_classes)
            except:
                continue
            all_batch_assd.append(one_case_assd)
    # Save all metrics
    all_dice = np.array(all_batch_dice)
    all_assd = np.array(all_batch_assd)
    np.save(results.joinpath("dice-per-volume.npy"), all_dice)
    np.save(results.joinpath("assd-per-volume.npy"), all_assd)
    # Display easy to read summary
    tab = Table(title="Evaluation metrics summary")
    tab.add_column("Metric")
    tab.add_column("Anterior leaflet", style="magenta")
    tab.add_column("Posterior leaflet", style="cyan")
    tab.add_column("Aggregate", justify="right", style="yellow")
    for name, metric in zip(["Dice", "ASSD"], [all_dice, all_assd]):
        if (metric == []).all(): #FIXME: Temporary patch to lack of ASSD 3D
            continue
        mean, std = metric.mean(axis=0), metric.std(axis=0)
        tab.add_row(name, f"{mean[0]} ± {std[0]}", f"{mean[1]} ± {std[1]}", f"{metric.mean()} ± {metric.std()}")
    console = Console(record=True)
    console.print(tab), console.save_text(results.joinpath("metrics-summary.txt"))



if __name__ == "__main__":
    test()
