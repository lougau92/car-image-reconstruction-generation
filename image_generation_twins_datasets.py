# genereate vehicle reconstructions images using the best model parameters found in the tuning experiment,
#  for the entire nrb dataset
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

main_path = "/home/p63744/projects/louis/cars_data/nrb_dataset/images/"
main_save_path ="/home/p63744/projects/louis/cars_data/nrb_dataset/twin_ds/images/"
prompt = "A picture of a car"
negative_prompt = "a damaged and broken car with broken glass and scratches"
best_params = {'strength': 0.6668866792315296, 'guidance': 43.06837407750286}

done = []

import numpy as np
from torchvision import transforms
from src.utils import utils
from  tqdm import tqdm



splits = ["train","val","test"]

for split in splits:

    path = main_path+split+"/"
    save_path = main_save_path+split+"/"

    
    for im in tqdm(os.listdir(path)):

        init_image = Image.open(path+im).convert("RGB")
        init_image = init_image.resize((768, 512))

        # generate a set of 4 images of the reconstruction of the vehicle
        for i in range(2):
            images = pipe(prompt=prompt,negative_prompt=negative_prompt, init_image=init_image, strength=best_params["strength"], guidance_scale=best_params["guidance"], num_images_per_prompt = 2).images
            for j in range(2):
                t = transforms.ToTensor()(images[j])
                # check if the image is nsfw, if so, regenerate it
                if torch.sum(t)==0:
                    print("nsfw image")
                    while torch.sum(t)==0:
                        images[j] = pipe(prompt=prompt,negative_prompt=negative_prompt, init_image=init_image, strength=best_params["strength"], guidance_scale=best_params["guidance"], num_images_per_prompt = 1).images[0]
                        t = transforms.ToTensor()(images[j])
                        print("changing image")

                im_file = save_path+str(2*i+j)+"_"+im
                images[j].save(im_file)
        
        # remove the worst 3 images
        scores = []
        for i in range(4):
            recon_im = str(i)+"_"+im
            im_score = utils.get_score(path+im,save_path+recon_im)
            scores.append((recon_im,im_score[0],im_score[1],im_score[2]))
        scores_sorted = sorted(scores, key= lambda x: x[1])
        print(scores_sorted)
        for i in range(3):
            print("remove",scores_sorted[i][0])
            os.remove(save_path+scores_sorted[i][0])
            