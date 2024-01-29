# Compression Benchmark Calculation

This folder consists of code that calculates the SSIM, PSNR, compression time and decompression time from original non-lossy png files with jpg, webp, heif, and avif files of different qualities ranging from 0 to 100.

## How to use

```exportDiffFormats.py``` takes non-lossy ```png``` files and compresses images as various image file extensions of different qualities in the ```output_folder```.

```bash
python 000_exportDiffFormats.py -d {image_directory} -o {output_folder} -e {image_extension}
```
