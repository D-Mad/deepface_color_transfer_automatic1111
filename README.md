
![image](https://github.com/D-Mad/deepface_color_transfer_automatic1111/assets/8207507/06ba06f3-b891-4058-9913-0ef29afee693)
![image](https://github.com/D-Mad/deepface_color_transfer_automatic1111/assets/8207507/674b2836-ba8f-4a45-a8c4-9248fb3db49c)






Color Transfer Extension for Automatic1111's Stable Diffusion
This repository contains an extension for AUTOMATIC1111's Stable Diffusion that allows you to perform color transfer between two images using OpenCV. The color transfer technique adjusts the color distribution of a source image to match the color distribution of a target image while preserving the original content of the source image.
Features

Load a source image and a target image
Transfer the color distribution from the target image to the source image
Adjust the intensity of color transfer
Adjust the balance of red, green, and blue colors
Adjust the saturation of colors
Adjust the contrast of the result image
Adjust the brightness and darkness of the result image

Requirements

AUTOMATIC1111's Stable Diffusion installed and running
OpenCV
NumPy
Gradio

Installation

Clone this repository into the extensions folder of your Stable Diffusion installation:

Copy codecd /path/to/stable-diffusion-webui/extensions
git clone https://github.com/your-username/color-transfer.git

Restart the Stable Diffusion web UI.

Usage

Open the Stable Diffusion web UI and navigate to the "Color Transfer" tab.
Upload or select a source image and a target image.
Adjust the sliders for intensity, color balance, saturation, contrast, brightness, and darkness.
Click the "Transfer Color" button to apply the color transfer to the source image.

Code Overview
The colortransfer.py script contains the following main components:

color_transfer function: This function performs the color transfer operation using the OpenCV library. It takes the source image, target image, and various adjustment parameters as input, and returns the result image.
on_ui_tabs function: This function sets up the Gradio interface with image inputs, sliders for adjustments, and a button to trigger the color transfer process.
script_callbacks.on_ui_tabs call: This line registers the on_ui_tabs function with the Gradio interface, allowing it to be displayed and interacted with.


