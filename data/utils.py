import pandas as pd
import numpy as np
import os
from PIL import Image
import einops

def get_framelist_neuttex_and_neutvert(directory):
    # Load frame list; ie, (segment, frame) pairs
    frame_list_path = f"{directory}/frame_list.csv"
    framelist = pd.read_csv(frame_list_path, dtype=str, sep=r",")
    
    framelist = framelist.loc[(framelist["seg_id"] != "EXP_Neutral_color_checker") & 
                              (framelist["seg_id"] != "EXP_hair_dynamics_no_hand_interaction") &
                              (framelist["seg_id"] != "EXP_hair_dynamics_hand_interaction") &
                              (framelist["seg_id"] != "EXP_neutral_hairline") &
                              (framelist["seg_id"] != "EXP_ear")]
    
    # Neutral conditioning
    neut_framelist = framelist.loc[framelist["seg_id"] == "EXP_neutral_peak"].values.tolist()
    neut_framelist.sort()
    neut_avgtex = None
    neut_vert = None
    
    for _, neut_frame in neut_framelist:
        if os.path.exists(f"{directory}/kinematic_tracking/registration_vertices/{int(neut_frame):06d}.npy"):
            verts = np.load(
                f"{directory}/kinematic_tracking/registration_vertices/{int(neut_frame):06d}.npy"
            )
        else:
            verts = None

        if os.path.exists(f"{directory}/uv_images/color/{int(neut_frame):06d}.avif"):
            tex = np.asarray(Image.open(f"{directory}/uv_images/color/{int(neut_frame):06d}.avif"))
            tex = einops.rearrange(tex, "h w c -> c h w").astype(np.float32)
        else:
            tex = None

        # NOTE(julieta) only load one since this might be causing OOM issues
        if tex is not None and verts is not None:
            neut_avgtex = tex
            neut_vert = verts
            break
        
    if neut_avgtex is None:
        raise ValueError("Not able to find any neutral average textures")
    if neut_vert is None:
        raise ValueError("Not able to find any neutral vertices")
    
    return framelist, neut_avgtex, neut_vert
    
def getitem(idx: int, framelist, cameras):
    segment_and_frame = framelist.iloc[idx // len(cameras)]
    segment: str = segment_and_frame.seg_id
    frame: str = segment_and_frame.frame_id
    camera = cameras[idx % len(cameras)]
    return segment, frame, camera
