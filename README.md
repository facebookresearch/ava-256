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

## Downloading the data
We provide a handy multithreaded script to download the dataset from AWS:

```bash
python download.py ava256/ -n 1 -j 1
```
Will download a single capture to a new folder, `ava256/`, using a single thread.
You may increase `n` to download more captures, and `-j` to increase the number of threads.

Run
```
python download.py --help
```
to see more download options.


## Tests
You can run tests with `python -m pytest tests/`

## Train
To train a simple model on a standalone machine you can run
```bash
python ddp-train.py config.py
```

To train on RSC, you can use
```bash
 ava sync oss-uca1; SCENV=ava rsc_launcher launch \
  --projects AIRSTORE_AVATAR_RSC_DATA_PIPELINE_CRYPTO \
  -e 'cd ~/rsc/oss-uca1 && \
  source /uca/conda-envs/activate-latest && \
  export GLOG_minloglevel=2 && \
  export NCCL_ASYNC_ERROR_HANDLING=1 && \
  export DB_CACHE_DIR=/shared/airstore_index/avatar_index_cache && \
  python ddp-train.py config.py'
```

## Visualization
To run visualization, you can run
```
python render.py
```

## Metrics
TODO

## TODOs before releasing

* Select 256 ids
* Train/val/test partitions
* Point to pre-trained model(s)

### Devex
* See TODOs in download script
* Pytorch lightning for multi-GPU?

### Models

* `mlp2d.py` ~~clean up~~, add tests
* `models/colorcals` ~~clean up~~, add tests

### Organization and design

* Maybe put codebase under `src/` folder?
* Organize tests into subfolders (need to figure out why imports break then?)
