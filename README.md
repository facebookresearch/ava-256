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
You can run tests with `pytest tests/`

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
TODO

## Metrics
TODO


## TODOs before releasing

* ~~Metrics in pixel space~~
* Train/val/test partitions
* See if we can include Yaser (PRW754) in the release
  * His last capture was Pilots in October 2022, which might be hard to port
  * https://www.internalfb.com/manifold/explorer/codec-avatars-captures-prod-es/tree/captures/v2/pilots/mugsy/Mar19/m--20221006--1031--PRW754--pilot--CA2FullBody--Heads
  * Yasser's capture is part of multiface (he's the mini-example you can download... maybe use that one?)

### Devex
* Script to download/update dataset if you already have a copy
* Make script to decide
* Figure out pip/conda install for the repo (maybe not? since it's not intended to be used as a library)
* We have both cv2 and PIL as dependencies. We should remove one of them (probably cv2, since I don't think we are using it for anything non-trivial)
* Write fast download script
* Download script
* Nice tensorboard, but not too intrusive with the main training loop
* Pytorch lightning for multi-GPU?


### Assets and asset provenance
* Should we use grown neck geometry? Results look much better and avoids some of the worst pathologies MVP
* I'm Using CARE's (Chenglei's?) `load_obj` function, because all the darn off the shelf libraries that can load obj
files have issues, see if we can switch to something else
* ~~I ported `make_closest_uv_barys` and its dependencies to this repo hoping I'd be able to re-compute the arguments~~
~~of the encoders (`bary_idx` and `bary_img` on the fly, instead of giving people the pre-computed images, since that's~~
~~north of 30MB~~
  * ~~:fire: `make_closest_uv_barys` takes ~38 seconds for a size of 1024 on my threadripper. There is something seriously wrong with this function and we should make it faster. Run `pytest tests/test_expression_encoder.py test_sizes` to repro the slowness~~ Fixed with a fast IGL function for distance computation
  * ~~I have not been able to fully reproduce the pre-computed `bary_img`, see `pytest tests/test_expression_encoder.py` for my failed attempts. We should either get a consistent repro, or test that the newly-computed image produces reasonable results~~ Verified that we can get reasonable results with on-the-fly assets
  * `Trimesh` and friends is a long list of dependencies (`rtree`, `scipy`), consider rewriting the whole thing


### More assets
* Add support for FLAME topology and meshes -- show that this is worse than our registered mesh

### Extensions
* ~~Swap the current `extensions` for `extension-mvp`, which are taken from [the MVP repo](https://github.com/facebookresearch/mvp/tree/main/extensions).~~
* ~~Add instructions for how to compile said extensions~~
* ~~Remove wuffs (`png_reader`) extension - I don't think it's used?~~

### Models

* `mlp2d.py` ~~clean up~~, add tests
* `models/colorcals` ~~clean up~~, add tests

### Organization and design

* Organize tests into subfolders (need to figure out why imports break then?)
* Get rid of config.py? Maybe use a proper configuration system like YACS?


## Open questions

* What does the structure for data look like?
   * Needs to be something that allows people to download subset of the data efficiently (eg, just meshes)
