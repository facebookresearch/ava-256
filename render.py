import torch
import pathlib
import argparse
from typing import Dict, List, Union
import numpy as np
import time
import einops
from PIL import Image
import yaml
from fvcore.common.config import CfgNode as CN

from data.ava_dataset import MultiCaptureDataset as AvaMultiCaptureDataset
from data.ava_dataset import SingleCaptureDataset as AvaSingleCaptureDataset
from data.ava_dataset import none_collate_fn
from data.utils import MugsyCapture
from utils import load_checkpoint, tocuda, get_autoencoder, render_img, train_csv_loader
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Cross ID driving")
    parser.add_argument("--checkpoint", type=str, default="run/aeparams.pt", help="checkpoint location")
    parser.add_argument("--output-dir", type=str, default="viz/", help="output image directory")
    parser.add_argument("--config", default="configs/config.yaml", type=str, help="config yaml file")
    
    # Cross ID visualization configuration
    parser.add_argument("--driver-id", type=str, default="20230324--0820--AEY864", help="id of the driver avatar")
    parser.add_argument("--driven-id", type=str, default="20230831--0814--ADL311", help="id of the driven avatar")
    parser.add_argument("--camera-id", type=str, default="401031", help="render camera id")
    parser.add_argument("--segment-id", type=str, default="EXP_cheek001", help="segment to render; render all available frames if None")
    parser.add_argument("--opts", default=[], type=str, nargs="+")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = CN(yaml.load(file, Loader=yaml.UnsafeLoader))

    config.merge_from_list(args.opts)

    train_params = config.train
    
    output_dir = args.output_dir + '/' + args.driver_id + "_" + args.driven_id + "+" + args.segment_id
    
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Train dataset mean/std texture and vertex for normalization
    train_captures, train_dirs = train_csv_loader(train_params.base_dir, train_params.data_csv, train_params.nids)
    dataset = AvaMultiCaptureDataset(train_captures, train_dirs, downsample=train_params.downsample)
    
    batchsize = 1
    numworkers = 1
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=False,
        drop_last=True,
        num_workers=numworkers,
        collate_fn=none_collate_fn,
    )

    # Get Autoencoder
    assetpath = pathlib.Path(__file__).parent / "assets"
    ae = get_autoencoder(dataset, assetpath=assetpath)

    # Load from checkpoint
    ae = load_checkpoint(ae, args.checkpoint).cuda()
    
    # Set to Evaluation mode
    ae.eval()    

    # Driver capture dataloader
    driver_capture = MugsyCapture(mcd=args.driver_id.split("--")[0], mct=args.driver_id.split("--")[1], sid=args.driver_id.split("--")[2])
    driver_dir = f"{train_params.base_dir}/{args.driver_id}/decoder"
    driver_dataset = AvaSingleCaptureDataset(driver_capture, driver_dir, downsample=train_params.downsample)
    
    # Driven capture dataloader
    driven_capture = MugsyCapture(mcd=args.driven_id.split("--")[0], mct=args.driven_id.split("--")[1], sid=args.driven_id.split("--")[2])
    driven_dir = f"{train_params.base_dir}/{args.driven_id}/decoder"
    driven_dataset = AvaSingleCaptureDataset(driven_capture, driven_dir, downsample=train_params.downsample)
    
    texmean = dataset.texmean
    vertmean = dataset.vertmean
    texstd = dataset.texstd
    vertstd = dataset.vertstd 
    
    # Delete dataset because it is no longer used
    del dataset

    # Grab driven normalization stats
    for dataset in [driver_dataset, driven_dataset]:
        dataset.texmean = texmean
        dataset.texstd = texstd
        dataset.vertmean = vertmean
        dataset.vertstd = vertstd
    
    # select only desired segments
    if args.segment_id:
        driver_dataset.framelist = driver_dataset.framelist.loc[driver_dataset.framelist["seg_id"] == args.segment_id]
    if driver_dataset.framelist.values.tolist() == []:
        raise ValueError(f"Asked to render Segment {args.segment_id}, but there are no frames with that Segment in {driver_capture}")

    # select only desired cameras
    if args.camera_id:
        if args.camera_id in driver_dataset.cameras:
            driver_dataset.cameras = [args.camera_id]
        else: 
            raise ValueError(f"Camera id {args.camera_id} is not defined for {driver_capture}")
    else:
        # TODO (Emily): Generalize choosing from frontal cameras i.e. ["401031", "401880", "401878"]
        driver_dataset.cameras = ["401031"]
        
    driver_loader = torch.utils.data.DataLoader(
        driver_dataset,
        batch_size=batchsize,
        shuffle=False,
        drop_last=False,
        num_workers=numworkers,
        collate_fn=none_collate_fn,
    )
    
    driven_loader = torch.utils.data.DataLoader(
        driven_dataset,
        batch_size=batchsize,
        shuffle=False,
        drop_last=False,
        num_workers=numworkers,
        collate_fn=none_collate_fn,
    )
    driveniter = iter(driven_loader)
    driven = next(driveniter)
    
    while driven is None:
        driven = next(driveniter)
    
    iter = 0
    
    for driver in tqdm(driver_loader, desc="Rendering Frames"):
        # Skip if any of the frames is empty
        if driver is None:
            continue
        
        cudadriver: Dict[str, Union[torch.Tensor, int, str]] = tocuda(driver)
        cudadriven: Dict[str, Union[torch.Tensor, int, str]] = tocuda(driven)

        running_avg_scale = False
        gt_geo = None
        residuals_weight = 1.0
        output_set = set(["irgbrec", "bg"])

        # Generate image from original inputs
        output_orig = ae(
                cudadriver["camrot"],
                cudadriver["campos"],
                cudadriver["focal"],
                cudadriver["princpt"],
                cudadriver["modelmatrix"],
                cudadriver["avgtex"],
                cudadriver["verts"],
                # normalized using the train data stats and driver data stats
                cudadriver["neut_avgtex"],
                cudadriver["neut_verts"], 
                cudadriver["neut_avgtex"],
                cudadriver["neut_verts"], 
                cudadriver["pixelcoords"],
                cudadriver["idindex"],
                cudadriver["camindex"],
                running_avg_scale=running_avg_scale,
                gt_geo=gt_geo,
                residuals_weight=residuals_weight,
                output_set=output_set,
            )
        
        # Generate image from cross id texture and vertex
        output_driven = ae(
                cudadriver["camrot"],
                cudadriver["campos"],
                cudadriver["focal"],
                cudadriver["princpt"],
                cudadriver["modelmatrix"],
                cudadriver["avgtex"],
                cudadriver["verts"],
                # normalized using the train data stats and driven data stats
                cudadriver["neut_avgtex"],
                cudadriver["neut_verts"], 
                cudadriven["neut_avgtex"],  
                cudadriven["neut_verts"], 
                cudadriver["pixelcoords"],
                cudadriver["idindex"],
                cudadriver["camindex"],
                running_avg_scale=running_avg_scale,
                gt_geo=gt_geo,
                residuals_weight=residuals_weight,
                output_set=output_set,
            )
        
        # Grab ground truth frame from the driver
        gt = cudadriver["image"].detach().cpu().numpy()
        gt = einops.rearrange(gt, "1 c h w -> h w c")
        
        rgb_orig = output_orig["irgbrec"].detach().cpu().numpy()
        rgb_orig = einops.rearrange(rgb_orig, "1 c h w -> h w c")
        
        rgb_driven = output_driven["irgbrec"].detach().cpu().numpy()
        rgb_driven = einops.rearrange(rgb_driven, "1 c h w -> h w c")
        
        render_img([[gt, rgb_orig, rgb_driven]], f"{output_dir}/img_{iter:06d}.png")

        iter += 1