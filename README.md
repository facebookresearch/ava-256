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
  python3 sbatch.py -n 1 -g 8 -t 1 -w 4 --source-dir /home/$USER/rsc/oss-uca1/ \
  --checkpoint-root-dir /checkpoint/avatar/$USER/oss_release/ \
  --batchsize 4 \
  --learning-rate 1e-4 \
  --masterport $(shuf -i 2049-65000 -n 1)'
```

## Visualization
To run visualization, you can run
```
python render.py
```

## Metrics
TODO

## TODOs before releasing

* Train/val/test partitions

### Devex
* See TODOs in download script
* Pytorch lightning for multi-GPU?

### Assets and asset provenance
* I'm Using CARE's (Chenglei's?) `load_obj` function, because all the darn off the shelf libraries that can load obj
files have issues, see if we can switch to something else

### Models

* `mlp2d.py` ~~clean up~~, add tests
* `models/colorcals` ~~clean up~~, add tests

### Organization and design

* Maybe put codebase under `src/` folder?
* Organize tests into subfolders (need to figure out why imports break then?)

## Dataset Issues

* Some subjects put their hair down, and this changes their appearance drastically. (20230831—-0814—-ADL311) Possible solution: Separate them as different IDs
* Some subjects have hands appear during capture (ie., 20230831—-0814—-ADL311 pulls hair down)
