# Ava-256: Universal Encoders and Decoders

### Part of Codec Avatar Studio

Ava-256 is designed to enable universal face encoders and decoders.

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

## Downloading the data
We provide a handy multithreaded script to download the dataset from AWS:

```bash
python download.py -o <output-dir> -n 1 -j 8
```
Will download a single capture to `<output-dir>`, using 8 threads.
You may increase `n` to download more captures (up to 256), and `-j` to increase the number of threads.
Note that, by default, this will download the `4TB` dataset. If you want longer or higher
quality data (at the expense of more storage), you may pass `--size {8,16,32}TB`.

## Train
To train a simple model on a standalone machine you can
1. Update the config file under `configs/config.yaml`, specially the `dataset_dir`ectory to point to the dataset
2. run `python ddp-train.py`

Note that you can override any parameter in the config file by passing flags on the cli, eg
```bash
python ddp-train.py --train.dataset_dir=<mydir>
```
will override the `train.dataset_dir` parameter in the config file.

To train on Avatar RSC, you can use
```bash
 ava sync ava-256; SCENV=ava rsc_launcher launch \
  --projects AIRSTORE_AVATAR_RSC_DATA_PIPELINE_CRYPTO \
  -e 'cd ~/rsc/ava-256 && sbatch sbatch.sh'
```

## Visualization
To run visualization of a trained model, you can run
```bash
python render.py
```

## Tests
You can run tests with `python -m pytest tests/`
