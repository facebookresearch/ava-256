# Compression Benchmark Calculation

This folder consists of code that calculates the SSIM, PSNR, compression time and decompression time from original non-lossy png files with jpg, webp, heif, and avif files of different qualities ranging from 0 to 100.

## How to use

```000_exportDiffFormats.py``` takes non-lossy ```png``` files and compresses images as various image file extensions of different qualities in the ```output_folder```.

```bash
python 000_exportDiffFormats.py -d {image_directory} -o {output_folder} -e {image_extension}
```

Then with ```001_getSize+PSNR_SSIM.py```, you can calculate the PSNR and the SSIM between the original non-lossy png files with the newly generated image files.

```002_plotGraphsSize+PSNR_SSIM.py``` saves plots for PSNR, SSIM, compression time, and decompression time.

```003_selectFrames.py``` selects 100 frames from the MultiFace dataset.

```004_sortByBPP.py``` sorts by BPP as symlinks.