# genereate vehicle reconstructions images using the best model parameters found in the tuning experiment,
# on a random sample of 30 images from the test set
import torch
from PIL import Image

from diffusers import StableDiffusionImg2ImgPipeline

import os

# load the pipeline
device = "cuda:1"
model_id_or_path = "./stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id_or_path,
    revision="fp16", 
    torch_dtype=torch.float16,
)

pipe = pipe.to(device)

path = "/home/p63744/projects/louis/cars_data/nrb_dataset/images/test/"

prompt = "A picture of a car"
negative_prompt = "a picture of a damaged and broken car with broken glass and scratches"
best_params = {'strength': 0.6668866792315296, 'guidance': 43.06837407750286}


import numpy as np
from tqdm import tqdm
from torchvision import transforms
# pick 30 random images from the test set
sample_set = np.random.choice(os.listdir(path), 30, replace=False)


for im in tqdm(sample_set):
    print("image: ", im)
    init_image = Image.open(path+im).convert("RGB")
    init_image = init_image.resize((768, 512))

    im_dir = ("/home/p63744/projects/louis/cars_data/nrb_dataset/twins_imgs/samples_"+im.split(".")[0])
    os.mkdir(im_dir)
    init_image.save(im_dir+"/0.jpg")

    # generate a total of 15 different images of the reconstruction of the vehicle
    for i in range(5):
        images = pipe(prompt=prompt,negative_prompt=negative_prompt, init_image=init_image, strength=best_params["strength"], guidance_scale=best_params["guidance"], num_images_per_prompt = 3).images
        for j in range(3):
            t = transforms.ToTensor()(images[j])
            # check if the image is nsfw, if so, regenerate it
            if torch.sum(t)==0:
                print("nsfw image")
                while torch.sum(t)==0:
                    images[j] = pipe(prompt=prompt,negative_prompt=negative_prompt, init_image=init_image, strength=best_params["strength"], guidance_scale=best_params["guidance"], num_images_per_prompt = 1).images
                    t = transforms.ToTensor()(transforms.ToPILImage()(images[j]))
                    print("changing image")

            images[j].save(im_dir+"/"+str(i*3+j+1)+".jpg")


 

