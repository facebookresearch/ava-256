# Ava-256: Universal Encoders and Decoders

### Together with [Goliath](https://github.com/facebookresearch/goliath), part of Codec Avatar Studio

We provide 
* 256 paired high-resolution dome and headset captures
* Code to build universal face encoders and decoders

![256_subjects](https://github.com/facebookresearch/ava-256/assets/3733964/622eb5af-6375-4f24-830d-a0025d7a7d23)

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

## Data
We provide a handy multithreaded script to download the dataset from AWS:

```bash
python download.py -o <output-dir> -n 1 -j 8
```
Will download a single capture to `<output-dir>`, using 8 threads.
You may increase `n` to download more captures (up to 256), and `-j` to increase the number of threads.
Note that, by default, this will download the `4TB` dataset. If you want longer or higher
quality data (at the expense of more storage), you may pass `--size {8,16,32}TB`.

### Decoder

For every subject, the decoder data includes 80 camera views with camera calibration, head pose in world coordinates,
registered mesh, 3d keypoints and semantic segmentations. 

https://github.com/facebookresearch/ava-256/assets/3733964/03864d10-1613-4041-bc1d-0584769c2764

### Encoder

For every subject, the encoder data consists of 5 infrared camera views captured from a Quest Pro.

https://github.com/facebookresearch/ava-256/assets/3733964/0f49b797-f44e-47af-88d9-626d67c8d189

For more details on the data format, see Question 4 under the [Composition Section](https://github.com/facebookresearch/ava-256/blob/main/DATASHEET.md#composition) of our datasheet.

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

This should create a visualization showcasing the consistent expression space of the decoder:



https://github.com/facebookresearch/ava-256/assets/3733964/41613355-00f1-479e-9c39-c8480192bbaa



## Tests
You can run tests with `python -m pytest tests/`

## License
ava-256 is CC BY-NC 4.0 licensed, as found in the LICENSE file.

## Citation

Ava-256 is a labour of love due to many people at Meta.
If you find our dataset or codebase useful, please cite our work:
```
@article{martinez2024codec,
  author = {Julieta Martinez and Emily Kim and Javier Romero and Timur Bagautdinov and Shunsuke Saito and Shoou-I Yu and Stuart Anderson and Michael Zollhöfer and Te-Li Wang and Shaojie Bai and Chenghui Li and Shih-En Wei and Rohan Joshi and Wyatt Borsos and Tomas Simon and Jason Saragih and Paul Theodosis and Alexander Greene and Anjani Josyula and Silvio Mano Maeta and Andrew I. Jewett and Simon Venshtain and Christopher Heilman and Yueh-Tung Chen and Sidi Fu and Mohamed Ezzeldin A. Elshaer and Tingfang Du and Longhua Wu and Shen-Chi Chen and Kai Kang and Michael Wu and Youssef Emad and Steven Longay and Ashley Brewer and Hitesh Shah and James Booth and Taylor Koska and Kayla Haidle and Matt Andromalos and Joanna Hsu and Thomas Dauer and Peter Selednik and Tim Godisart and Scott Ardisson and Matthew Cipperly and Ben Humberston and Lon Farr and Bob Hansen and Peihong Guo and Dave Braun and Steven Krenn and He Wen and Lucas Evans and Natalia Fadeeva and Matthew Stewart and Gabriel Schwartz and Divam Gupta and Gyeongsik Moon and Kaiwen Guo and Yuan Dong and Yichen Xu and Takaaki Shiratori and Fabian Prada and Bernardo R. Pires and Bo Peng and Julia Buffalini and Autumn Trimble and Kevyn McPhail and Melissa Schoeller and Yaser Sheikh},
  title = {{Codec Avatar Studio: Paired Human Captures for Complete, Driveable, and Generalizable Avatars}},
  year = {2024},
  journal = {NeurIPS Track on Datasets and Benchmarks},
}
```
