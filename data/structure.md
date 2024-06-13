

<!-- TODO(julieta) Update with latest structure before release -->

## Proposed data and asset structure for ava-256

Top level folder

256 directories with

```
m--{capture.mcd}--{capture.mct}--{capture.sid.upper()}--GHS
.
└── decoder
    ├── camera_calibration.json
    ├── frame_list.csv
    ├── head_pose
    │   └── head_pose.zip
    ├── image
    │   ├── cam{camname}.zip
    │   ├── cam400943.zip
│   ├── mesh
│   │   ├── EXP_A.zip
│   │   └── EXP_B.zip
│   ├── tex_mean.png
│   ├── tex_var.txt
│   ├── unwrapped_tex
│   ├── vert_mean.bin
│   └── vert_var.txt
└── encoder
```

7 directories, 12 files
