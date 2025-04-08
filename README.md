# Image Deconvolution Tool
A Python-based tool for performing image deconvolution (with spatial blur) using Wiener or Richardson-Lucy techniques. The image is assumed to have a Moffat PSF (Point Spread Function) blur.

# This repo is not finalized yet ! [april 8, 10am] 

## Installation
### Prerequisites
- Python 3.8+
- `pip` installed

### Steps
1. Clone the repository:
   ```sh
   git clone https://github.com/ryadguezzi/astro_deconvo.git
   cd deconvolution

2. Install dependencies
   pip install -r requirements.txt

3. Check that the project is working correctly
    Run 
    ```sh
    python3 ./deconvolution_method/main.py images/original/star_field_100.jpg

## Usage
Run the main script:
``sh
python3 ./deconvolution_method/main.py <input_image.png> <find_parameters>

## Features
This project has two main functionalities:
- Given an original image, it can spatially blur the image and then deblur it using the Wiener or Richardson-Lucy deconvolution method, using default values for the parameters alpha and beta of the Moffat function.
- Given an original image, it can spatially blur the image, find approximate values for the alpha and beta parameters of the Moffat function, and then deblur it using the Wiener or Richardson-Lucy deconvolution method.
