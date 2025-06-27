import json
import random

import pickle
import numpy as np
import gradio as gr
from PIL import Image

from api import get_prompt_images
from settings import COMFYUI_PATH


STYLE_LIST = [
    {"name": "Cinematic",
    "prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy"},
    {"name": "3D Model",
    "prompt": "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting"},
    {"name": "Anime",
    "prompt": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed"},
    {"name": "Digital Art",
    "prompt": "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed"},
    {"name": "Photographic",
    "prompt": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed"},
    {"name": "Pixel art", "prompt": "pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics"},
    {"name": "Fantasy art",
    "prompt": "ethereal fantasy concept art of  {prompt} . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy"},
    {"name": "Neonpunk",
    "prompt": "neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional"},
]


def get_styled_prompt(style_name: str, base_prompt: str) -> str:
    for style in STYLE_LIST:
        if style["name"].lower() == style_name.lower():
            return style["prompt"].replace("{prompt}", base_prompt)
    raise ValueError(f"Style '{style_name}' not found.")


def save_input_image(image):
    input_image = f"{COMFYUI_PATH}/input/sample_sketch.png"
    with open('./data.p', 'wb') as fp:
        pickle.dump(image, fp, protocol=pickle.HIGHEST_PROTOCOL)
    pillow_image = Image.fromarray(np.array(image["composite"]))
    pillow_image.save(input_image)


def process(
    positive_prompt, 
    image,
    style,
    seed,
    guidance
    ):
    with open("workflow.json", "r", encoding="utf-8") as f:
        prompt = json.load(f)

    prompt["3"]["inputs"]["seed"] = seed
    prompt["13"]["inputs"]["strength"] = guidance
    prompt["6"]["inputs"]["text"] = get_styled_prompt(style, positive_prompt)

    save_input_image(image)
    images = get_prompt_images(prompt)
    return images


demo = gr.Interface(
    fn=process,
    inputs=[
        # prompt
        gr.Textbox(label="Positive Prompt"),
        # sketch image
        gr.Sketchpad(
            type="pil",
            height=512,
            width=512,
            min_width=512,
            image_mode="RGBA",
            show_label=False,
            mirror_webcam=False,
            show_download_button=True,
            elem_id='input_image',
            brush=gr.Brush(colors=["#000000"], color_mode="fixed", default_size=4),
            canvas_size=(1024, 1024),
            layers=False
        ),
        # style
        gr.Dropdown(
            label="Style",
            choices=[style["name"] for style in STYLE_LIST],
            value="Cinematic",
            scale=1,
        ),
        # seed
        gr.Textbox(label="Seed", value='42', scale=1, min_width=50),
        # guidance
        gr.Slider(
            label="Sketch guidance",
            show_label=True,
            minimum=0,
            maximum=1,
            value=0.4,
            step=0.01,
            scale=3,
        )

    ],
    outputs=[gr.Gallery(label="Result")],
)


if __name__ == "__main__":

    demo.queue()
    demo.launch()