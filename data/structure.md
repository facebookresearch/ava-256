

<!-- TODO(julieta) Update with latest structure before release -->

## Proposed data and asset structure for ava-256

Top level folder

256 directories with

```
m--{capture.mcd}--{capture.mct}--{capture.sid.upper()}--GHS
.
├── decoder
│   ├── KRT
│   ├── frame_list.csv
│   ├── images
│   │   ├── EXP_A
│   │   │   ├── CAM1.zip
│   │   │   └── CAM2.zip
│   │   └── EXP_B
│   │       ├── CAM1.zip
│   │       └── CAM2.zip
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
