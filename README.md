# Universal Decoders

You can use this code together with the [encoder release](TODO-link) to create universal encoders and decoders.


This code was written by Julieta Martinez, Jason Saraghi, Javier Romero, and ???

If you find this code or the accompanying data useful, please cite our work
```
TODO(julieta) bibtex here
```

## Compiling extensions
We need to compile the CUDA raymarcher and some utilities. This can be done with

```bash
cd extensions/mvpraymarch
make
```
and
```bash
cd extensions/utils
make
```

## Dataset

We provide Dome and Quest Pro captures for 256 subjects.

### Decoder (Dome) captures

These captures consist of 80 cameras each, each stored in RGB.

| Dataset size   | Image resolution  |  FPS   | Frames per capture  | AVIF quality |
| -------------: | ----------------: | -----: | ------------------: | -----------: |
|  4 TB          | 1024 x  667       |  7.5   |  ~ 5_000            | 63           |
|  8 TB          | 1024 x  667       | 15.0   | ~ 10_000            | 63           |
| 16 TB          | 2048 x 1334       |  7.5   |  ~ 5_000            | 70           |  <!-- TODO(julieta): Quality could be higher, we still have space -->
| 32 TB          | 2048 x 1334       | 15.0   | ~ 10_000            | 70           |  <!-- TODO(julieta): Quality could be higher, we still have space -->

The total number of images is roughly 400k - 800k per capture, depending on the dataset size.

### Encoder (Quest Pro) captures

These captures consist of 5 infrared (IR) cameras each

| Image resolution  |  FPS   | Frames per capture  | AVIF quality |
| ----------------: | -----: | ------------------: | -----------: |
| 400 x 400         | TBD    |  ~ 10_000           | TBD          |


## Downloading the data
We provide a handy multithreaded script to download the dataset from AWS:

```bash
python download.py ava256/ -n 4 -j 8
```
Will download a single capture to a new folder, `ava256/`, using 8 threads.
You may increase `n` to download more captures, and `-j` to increase the number of threads.
Note that, by default, this will download the `4TB` dataset. If you want longer or higher
quality data (at the expense of more storage), you may pass `--size {8,16,32}TB`.

Run
```bash
python download.py --help
```
to see more download options.

## Train
To train a simple model on a standalone machine you can
1. Update the config file under `configs/config.yaml`, specially the `dataset_dir`ectory to point to the dataset
2. run `python ddp-train.py`

Note that you can override any parameter in the config file by passing flags on the cli, eg
```bash
python ddp-train.py --train.dataset_dir=<mydir>
```
will override the `train.dataset_dir` parameter in the config file.


<!--NOTE(julieta) delete before releasing -->

To train on Avatar RSC, you can use
```bash
 ava sync ava-256; SCENV=ava rsc_launcher launch \
  --projects AIRSTORE_AVATAR_RSC_DATA_PIPELINE_CRYPTO \
  -e 'cd ~/rsc/ava-256 && sbatch sbatch.sh'
```

## Visualization
To run visualization, you can run
```bash
python render.py
```

### License

See LICENSE.

## Metrics
TODO

## Tests
You can run tests with `python -m pytest tests/`

## TODOs before releasing
* Train/val/test partitions
* Point to pre-trained model(s)

## Nice to haves
* Allow training on multiface
* Unify losses and metrics with Goliath-4

### Devex
* See TODOs in download script -- hashes, resuming downloads, etc
* Pytorch lightning for multi-GPU (?)

### Models

* `mlp2d.py` ~~clean up~~, add tests
* `models/colorcals` ~~clean up~~, add tests

### Organization and design

* Maybe put codebase under `src/` folder?
* Organize tests into subfolders (need to figure out why imports break then?)
