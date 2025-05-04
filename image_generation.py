# generate samples of generated vehicle reconstructions at different strengths and guidance scales of the diffusion model
import torch
from PIL import Image

from diffusers import StableDiffusionImg2ImgPipeline


# load the pipeline
device = "cuda:1"
model_id_or_path = "./stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id_or_path,
    revision="fp16", 
    torch_dtype=torch.float16,
)

pipe = pipe.to(device)

# original image
path = "/home/p63744/projects/louis/cars_data/nrb_dataset/images/train/id_1626700207984.jpg"
init_image = Image.open(path).convert("RGB")


init_image.save("damaged_car.png")

import numpy as np
init_image = init_image.resize((768, 512))

# defining the hyperparameter values to try
strengths = np.linspace(0.3,0.65,5)
guidances = np.linspace(7,40,5)

for strength in strengths:
    for guidance in guidances:
        print(strength,guidance)
        prompt = "A picture of a car"
        negative_prompt = "a picture of a damaged and broken car with broken glass and scratches"
        images = pipe(prompt=prompt,negative_prompt=negative_prompt, init_image=init_image, strength=strength, guidance_scale=guidance, num_images_per_prompt = 4).images

        for i in range(4):
            images[i].save("/home/p63744/projects/louis/damaged-cars-assessor/images/twins/"+str(i)+"/whole_car_"+str(strength)+"_"+str(guidance)+".png")
            