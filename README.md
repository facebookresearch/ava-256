# Universal Decoders

You can use this code together with the


This code was written by Julieta Martinez, Jason Saraghi, Javier Romero, and ???

If you find this code or the accompanying data useful, please cite our work
```
TODO(julieta) bibtex here
```


# TODOs before releasing

* Script to download/update dataset if you already have a copy
* Metrics in pixel space
* Train/val/test partitions
* See if we can include Yaser (PRW754) in the release
    * His last capture was Pilots in October 2022, which might be hard to port
    * https://www.internalfb.com/manifold/explorer/codec-avatars-captures-prod-es/tree/captures/v2/pilots/mugsy/Mar19/m--20221006--1031--PRW754--pilot--CA2FullBody--Heads
    * Yasser's capture is part of multiface (he's the mini-example you can download... maybe use that one?)
* Should we use grown neck geometry? Results look much better and avoids some of the worst pathologies MVP
* I'm Using CARE's (Chenglei's?) `load_obj`` function, because all the darn off the shelf libraries have issues,
  see if we can switch to something else by default
* ibid for `make_closes_uv_barys` and functions that this depends on. Alternative is having bary images pre-computed, but at high res that's ~30MB
  * Ok Trimesh and friends is a long list of dependencies (rtree, scipy), consider rewriting the whole thing