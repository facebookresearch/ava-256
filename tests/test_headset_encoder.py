# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from models.headset_encoders.universal import UniversalEncoder


def test_encoder_multi_cond():
    """ Tests for the universal headset encoder with different number of conditionings. """
    in_chans = 1
    out_chans = 256
    num_views = 4

    # HMC input image: [B, Views, C, H, W]
    hmc_cam_imgs = torch.rand((2, num_views, in_chans, 400, 400), dtype=torch.float32)

    # Test 2 cond
    num_conds = 2
    hmc_cond_imgs = torch.rand((2, num_views, num_conds, in_chans, 400, 400), dtype=torch.float32)
    expression_encoder = UniversalEncoder(in_chans=in_chans, out_chans=out_chans, num_views=num_views, num_conditions=num_conds)
    expr_code = expression_encoder(hmc_cam_imgs, hmc_cond_imgs)["expression"]
    assert expr_code.shape == torch.Size([2, 256])

    # Test 1 cond
    num_conds = 1
    hmc_cond_imgs = torch.rand((2, num_views, num_conds, in_chans, 400, 400), dtype=torch.float32)
    expression_encoder = UniversalEncoder(in_chans=in_chans, out_chans=out_chans, num_views=num_views, num_conditions=num_conds)
    expr_code = expression_encoder(hmc_cam_imgs, hmc_cond_imgs)["expression"]
    assert expr_code.shape == torch.Size([2, 256])

    # Test 0 cond
    num_conds = 0
    # Create expression encoder
    expression_encoder = UniversalEncoder(in_chans=in_chans, out_chans=out_chans, num_views=num_views, num_conditions=num_conds)
    expr_code = expression_encoder(hmc_cam_imgs)["expression"]
    assert expr_code.shape == torch.Size([2, 256])
