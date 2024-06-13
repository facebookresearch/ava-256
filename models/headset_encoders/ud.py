from typing import Any, Dict, List, Optional, Union, Sequence
import numpy as np
import torch
import torch.distributed as dist

from utils import get_autoencoder, load_checkpoint
from models.headset_encoders.tools import rvec_to_R


class FixedFrontalViewGenerator(torch.nn.Module):
    """
    Generate frontal view camera parameters given batch information
    """
    def __init__(self, down_sample_factor: int = 2) -> None:
        super().__init__()
        # frontal view camera parameters
        self.down_sample_factor: int = down_sample_factor

        self.frontal_render_shape: Sequence[int] = [2048 // 2, 1334 // 2]
        self.register_buffer(
            "frontal_cam_pose", torch.tensor([[115.3817, 26.1830, 984.4196]]),
            persistent=False
        )  # 1 x 3
        self.register_buffer(
            "frontal_cam_rot",
            torch.tensor(
                [
                    [
                        [0.9913, 0.0173, -0.1303],
                        [0.0244, -0.9983, 0.0531],
                        [-0.1291, -0.0558, -0.9901],
                    ]
                ]
            ),
            persistent=False,
        )  # 1 x 3 x 3
        self.register_buffer(
            "frontal_focal",
            torch.tensor([[[5068.6011, 0.0000], [0.0000, 5063.9287]]]) / 2,
            persistent=False,
        )  # 1 x 2 x 2
        self.register_buffer(
            "frontal_princpt", torch.tensor([[751.2227, 967.2575]]) / 2,
            persistent=False
        )  # 1 x 2

    def forward(
        self,
        batch_size: int,
        randomize: bool,
        rvec: Optional[torch.Tensor] = None
    ) -> Dict[str, Union[int, torch.Tensor]]:
        """
        Generate a frontal view camera parameters for a specific batch size
        """
        dsf = self.down_sample_factor
        camera_params = {
            "camrot": self.frontal_cam_rot.expand(batch_size, -1, -1),
            "campos": self.frontal_cam_pose.expand(batch_size, -1),
            "focal": self.frontal_focal.expand(batch_size, -1, -1) / dsf,
            "princpt": self.frontal_princpt.expand(batch_size, -1) / dsf,
            "render_h": int(self.frontal_render_shape[0] // dsf),
            "render_w": int(self.frontal_render_shape[1] // dsf),
        }
        # Apply rotation to the view
        if rvec is not None:
            R = rvec_to_R(rvec)
            camera_params["camrot"] = camera_params["camrot"] @ R
            camera_params["campos"] = (camera_params["campos"].unsqueeze(1) @ R).squeeze(1)

        return camera_params


class UDWrapper(torch.nn.Module):
    """
    Universal Decoder for Codec Avatar that supports batch-based inference
    for even different subjects.

    Some key variables:
        - `exp_name`: The specific name/run of the universal decoder
        - `ident`: The set of identities that this avatar set contains
    """
    conditioning_fn_templates = ["{exp_name}/identity_conditioning/{ident}/id_cond.pkl"]
    ae_info_fn_template = "{exp_name}/ae_info.pkl"
    ae_params_fn_template = "{exp_name}/aeparams.pt"

    def __init__(
        self,
        ud_exp_name: str,
        identities: List[str]
    ) -> None:
        super().__init__()
        # Avoid adding as buffers.
        self.idents = identities
        self.ident_idx: Dict[str, int] = {ident: idx for idx, ident in enumerate(identities)}

        print("Loading identity conditioning")
        ident_cond = []
        for template in self.conditioning_fn_templates:
            fns = [template.format(exp_name=ud_exp_name, ident=ident) for ident in identities]
            ident_cond = []
            for fn in fns:
                try:
                    ident_cond.append(torch.load(fn, map_location=f"cuda:{dist.get_rank() if dist.is_initialized() else 0}"))
                except FileNotFoundError:
                    ident_cond.append(None)

            if all(x is None for x in ident_cond):
                # Check the next file path template
                continue

            if any(x is None for x in ident_cond):
                missing_idents = [
                    ident for ident, cond in zip(identities, ident_cond) if cond is None
                ]
                raise FileNotFoundError(
                    f"No available conditioning data for {', '.join(missing_idents)}"
                )
            break
        else:
            raise FileNotFoundError("No available conditioning data available.")

        print("Done loading identity conditioning")
        self.iden_b_cond_len: int = len(ident_cond[-1]["b_geo"])

        self.register_buffer("z_geo", torch.cat([cond["z_geo"] for cond in ident_cond], 0))
        self.register_buffer("z_tex", torch.cat([cond["z_tex"] for cond in ident_cond], 0))

        for idx in range(self.iden_b_cond_len):
            self.register_buffer(
                f"b_geo_{idx}", torch.cat([cond["b_geo"][idx] for cond in ident_cond], 0)
            )
            self.register_buffer(
                f"b_tex_{idx}", torch.cat([cond["b_tex"][idx] for cond in ident_cond], 0)
            )

        # Get ae_info
        # FIXME: Serialized by torch, must load by torch
        ae_info = torch.load(
            self.ae_info_fn_template.format(exp_name=ud_exp_name), 
            map_location=f"cuda:{dist.get_rank() if dist.is_initialized() else 0}"
        )

        # Get Autoencoder and load from checkpoint
        ae = get_autoencoder(ae_info, assetpath="assets").eval()
        ae = load_checkpoint(ae, self.ae_params_fn_template.format(exp_name=ud_exp_name)).cuda()

        self.ae: torch.nn.Module = ae
        self.ae.eval()
        self.ae.requires_grad_(False)
        self.register_buffer("modelmatrix", torch.eye(4).unsqueeze(0))

    def randomize_idents(self, n: int) -> List[str]:
        """
        Randomly select a set of identities from the available ones.
        """
        return list(np.random.choice(self.idents, n))

    def get_id_cond(self, idents: List[str], device: torch.device) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Collect the identity conditioning for a set of identities.
        """
        ident_indices = [self.ident_idx[ident] for ident in idents]
        id_cond_z = {
            "z_geo": self.z_geo[ident_indices].to(device, non_blocking=True), 
            "z_tex": self.z_tex[ident_indices].to(device, non_blocking=True)
        }
        id_cond_b = {
            "b_geo": [],
            "b_tex": [],
        }
        for i in range(self.iden_b_cond_len):
            id_cond_b["b_geo"].append(getattr(self, f"b_geo_{i}")[ident_indices].to(device, non_blocking=True))
            id_cond_b["b_tex"].append(getattr(self, f"b_tex_{i}")[ident_indices].to(device, non_blocking=True))
        id_cond = dict(**id_cond_b, **id_cond_z)
        return id_cond

    def decode_geo(self, encodings: torch.Tensor, idents: Optional[List[str]] = None) -> torch.Tensor:
        """
        Decode the facial geometry from a set of encodings.
        """
        if idents is None:
            idents = self.randomize_idents(encodings.shape[0])
        id_cond = self.get_id_cond(idents)
        return self.ae.decode_geo(id_cond=id_cond, encoding=encodings.reshape(-1, 16, 4, 4))

    def render(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Inference universal avatars by rendering their frontals

        Args:
            inputs: A dictionary containing the following keys: 
              - "encodings": a tensor of shape [bsz, n_latent_dim]
              - "idents": a list of identities (string)
              - "camera_params": a dictionary containing camera parameters
              - "alpha_mask" (optional)
        """
        camera_params = inputs["camera_params"]
        render_h = camera_params["render_h"]
        render_w = camera_params["render_w"]
        campos = camera_params["campos"]
        camrot = camera_params["camrot"]
        focal = camera_params["focal"]
        princpt = camera_params["princpt"]

        encodings = inputs["encodings"]
        idents = inputs["idents"]
        alpha_mask = inputs.get("alpha_mask", None)

        device = campos.device
        b = campos.size(0)

        if (
            not hasattr(self, "pixelcoords")
            or self.pixelcoords.size(2) != render_h
            or self.pixelcoords.size(3) != render_w
        ):
            px, py = np.meshgrid(
                np.arange(render_w).astype(np.float32), np.arange(render_h).astype(np.float32)
            )
            # 1 * h * w * 2
            self.pixelcoords = torch.from_numpy(np.stack((px, py), axis=-1)).unsqueeze(0).to(device)

        id_cond = self.get_id_cond(idents, device=encodings.device)
        cudadata = {
            "camrot": camrot.contiguous(),
            "campos": campos.contiguous(),
            "focal": focal.diagonal(dim1=1, dim2=2).contiguous(),
            "princpt": princpt.contiguous(),
            "modelmatrix": self.modelmatrix.expand(b, -1, -1).contiguous(),
            "pixelcoords": self.pixelcoords.expand(b, -1, -1, -1).contiguous(),
            "alpha_mask": alpha_mask,
        }

        decout = self.ae.decode(
            **cudadata,
            output_set=["irgbrec", "ialpha", "verts"],
            id_cond=id_cond,
            expr_encoding=encodings.reshape(b, 16, 4, 4),
        )

        return {
            "img_pred": decout["irgbrec"],
            "mask": decout["ialpha"].detach(),
            "verts": decout["verts"],
        }


if __name__ == "__main__":
    import time

    test_identities = [
        "XUD838_20230811--0000",
        "YYA788_20230718--0000"
    ]

    start = time.perf_counter()
    uca1 = UDWrapper(
        ud_exp_name="/uca/leochli/oss/ava256_universal_decoder",
        identities=test_identities
    )
    end = time.perf_counter()
    print(f"Constructing model with {len(test_identities)} ids, takes: {end - start}s.")

    view_generator = FixedFrontalViewGenerator(down_sample_factor=2)

    device = torch.device("cuda")
    uca1.to(device)
    view_generator.to(device)

    test_batch_size = 2

    with torch.no_grad():
        # Camera params
        camera_params = view_generator(test_batch_size, randomize=False)
        decoder_inputs = {
            "idents": test_identities,
            "encodings": torch.zeros((test_batch_size, 256), device=device),
            "camera_params": camera_params,
            "alpha_mask": torch.ones((1024, 1024), device=device),
        }

        outputs = uca1.render(decoder_inputs)

        rgb_img = outputs["img_pred"].cpu().permute(1, 2, 0, 3).flatten(2, 3).numpy()
        from PIL import Image
        print(rgb_img.shape)
        Image.fromarray(rgb_img.transpose(1, 2, 0).astype(np.uint8)).save("/uca/shaojieb/test_ava256.png")
