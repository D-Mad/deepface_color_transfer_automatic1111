import modules.scripts as scripts
import gradio as gr
import os
import cv2
import numpy as np
from modules import script_callbacks

def color_transfer(source_image, target_image, intensity, red_balance, green_balance, blue_balance, saturation, contrast, brightness, darkness):
    source = cv2.cvtColor(np.array(source_image), cv2.COLOR_RGB2LAB)
    target = cv2.cvtColor(np.array(target_image), cv2.COLOR_RGB2LAB)

    s_mean, s_std = cv2.meanStdDev(source)
    t_mean, t_std = cv2.meanStdDev(target)

    s_mean = np.hstack(np.around(s_mean, 2))
    s_std = np.hstack(np.around(s_std, 2))
    t_mean = np.hstack(np.around(t_mean, 2))
    t_std = np.hstack(np.around(t_std, 2))

    result = source.copy()
    height, width, channel = source.shape

    for i in range(0, height):
        for j in range(0, width):
            for k in range(0, channel):
                x = source[i, j, k]
                x = ((x - s_mean[k]) * (t_std[k] / s_std[k])) + t_mean[k]
                x = round(x)
                x = 0 if x < 0 else x
                x = 255 if x > 255 else x
                result[i, j, k] = x

    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    # Adjust intensity
    result = cv2.addWeighted(np.array(source_image), 1 - intensity, result, intensity, 0)

    # Adjust color balance
    result = result.astype(np.float32)
    result[:, :, 0] *= red_balance
    result[:, :, 1] *= green_balance
    result[:, :, 2] *= blue_balance
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Adjust saturation
    result = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
    result[:, :, 1] = result[:, :, 1] * saturation
    result = cv2.cvtColor(result, cv2.COLOR_HSV2RGB)

    # Apply contrast adjustment
    result = cv2.addWeighted(result, contrast, np.zeros_like(result), 1 - contrast, 0)

    # Apply brightness adjustment
    result = cv2.addWeighted(result, 1, np.zeros_like(result), brightness / 100, brightness / 100 * 255)

    # Apply darkness adjustment
    result = cv2.addWeighted(result, 1, np.zeros_like(result), 1, darkness / 100 * -255)

    return result

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as color_transfer_interface:
        with gr.Row():
            source_image = gr.Image(label="Source Image", type="pil", height=256, width=256)
            target_image = gr.Image(label="Target Image", type="pil", height=256, width=256)
        with gr.Row():
            result_image = gr.Image(label="Result Image", type="pil", height=512, width=512)
        with gr.Row():
            intensity_slider = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.5, label="Intensity")
            red_balance_slider = gr.Slider(minimum=0.5, maximum=1.5, step=0.01, value=1.0, label="Red Balance")
            green_balance_slider = gr.Slider(minimum=0.5, maximum=1.5, step=0.01, value=1.0, label="Green Balance")
            blue_balance_slider = gr.Slider(minimum=0.5, maximum=1.5, step=0.01, value=1.0, label="Blue Balance")
            saturation_slider = gr.Slider(minimum=0.5, maximum=1.5, step=0.01, value=1.0, label="Saturation")
            contrast_slider = gr.Slider(minimum=0.5, maximum=1.5, step=0.01, value=1.0, label="Contrast")
            brightness_slider = gr.Slider(minimum=-100, maximum=100, step=1, value=0, label="Brightness")
            darkness_slider = gr.Slider(minimum=-100, maximum=100, step=1, value=0, label="Darkness")
        with gr.Row():
            transfer_button = gr.Button("Transfer Color")
            transfer_button.click(
                color_transfer,
                inputs=[source_image, target_image, intensity_slider, red_balance_slider, green_balance_slider, blue_balance_slider, saturation_slider, contrast_slider, brightness_slider, darkness_slider],
                outputs=[result_image],
            )

    return [(color_transfer_interface, "Color Transfer", "color_transfer_tab")]

script_callbacks.on_ui_tabs(on_ui_tabs)
