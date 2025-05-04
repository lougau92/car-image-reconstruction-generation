import torchvision
from PIL import Image
# from torchvision.io import read_image
# from torchvision.models import resnet50,efficientnet_v2_s, EfficientNet_V2_S_Weights, ResNet50_Weights
# from skimage.metrics import structural_similarity as ssim
import os
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from torchvision import  models
import torch
import torch.nn as nn
import numpy as np
import cv2
import torchvision.transforms as transforms
from transformers import ViTForImageClassification


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft = nn.Sequential(model_ft, nn.Sigmoid())
model_ft = model_ft.to(device)
car_model_path = "/home/p63744/projects/louis/damaged-cars-assessor/models/Car_classifiers/model.pth"
model_ft.load_state_dict(torch.load(car_model_path))
model_ft.eval()

model_cd = ViTForImageClassification.from_pretrained('./pytorch_model.bin',config='./config_car_embedder.json')
model_cd.classifier = torch.nn.Identity(768)
model_cd.to(device)
model_cd.eval()

transform = transforms.Compose([
    torchvision.transforms.Resize((224,224), interpolation = torchvision.transforms.InterpolationMode.BICUBIC ),
    transforms.ToTensor()
])

def get_diff_score(im1,im2,model):   
    pair = torch.cat((im1.unsqueeze(0),im2.unsqueeze(0)),axis =0)
    emds = model(pair)[0]
    mae = sum(abs(emds[0]-emds[1]))
    # print("MAE",round(float(mae,2))
    reg = torch.max(sum(abs(emds[0])),sum(abs(emds[1])))
    mae_reg = float(mae/reg)
    # print("MAE normalised",round(mae_reg,2))
    return mae_reg


def get_score(im_path,damaged_im_path, car_weight = 1, diff_weight = 1):


    recon_img = Image.open(im_path)
    recon_img = transform(recon_img)
    recon_img = recon_img.to(device)
    if torch.sum(recon_img)==0: return None
    
    damaged_img = Image.open(damaged_im_path)
    damaged_img = transform(damaged_img)
    damaged_img = damaged_img.to(device)

    car_classifier =True
    car_diff = True
    ssim = False
    
    car_diff_score =0

    if car_diff:
        with torch.no_grad():
            car_diff_score  = get_diff_score(recon_img,damaged_img,model_cd) 
   
    if ssim:
        damaged_im = cv2.imread(damaged_im_path)
        damaged_im = cv2.cvtColor(damaged_im, cv2.COLOR_BGR2GRAY)
        recon = cv2.resize(recon, (damaged_im.shape[1],damaged_im.shape[0]) ,interpolation	= cv2.INTER_LANCZOS4)
        recon_gray = cv2.cvtColor(recon, cv2.COLOR_BGR2GRAY)
    
        car_diff_score = ssim(damaged_im,recon_gray)

    car_score = 0

    if car_classifier:
        with torch.no_grad():
                outputs = model_ft(recon_img.unsqueeze(0))
                # print(outputs)
                car_score = float(outputs[0][0])
                

    tot_score = car_weight* car_score - diff_weight* car_diff_score
    if car_score < 0.65: tot_score -=1
    
    return (tot_score,car_diff_score,car_score)

from diffusers import StableDiffusionImg2ImgPipeline
import numpy as np
import torch
from PIL import Image
import shutil

damaged_im_path = "/home/p63744/projects/louis/damaged-cars-assessor/damaged_car.png"
p = "/home/p63744/projects/louis/damaged-cars-assessor/images/tune_twin/"


try:
    shutil.rmtree(p)
except Exception as e: 
    print(e)
os.mkdir(p)
os.mkdir(p+"labels/")

# https://github.com/huggingface/diffusers#text-to-image-generation-with-stable-diffusion
device = "cuda"
model_id_or_path = "./stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id_or_path,
    revision="fp16", 
    torch_dtype=torch.float16,
)
pipe = pipe.to(device)
init_image = Image.open(damaged_im_path).convert("RGB")
init_image = init_image.resize((768, 512), Image.Resampling.LANCZOS)
# pipe.enable_attention_slicing() #if ram issues
prompt = "A picture of a car"
negative_prompt = "a picture of a damaged and broken car with broken glass and scratches"

import optuna
torch.cuda.empty_cache()
from optuna.visualization import plot_contour

def objective(trial):

    strength = trial.suggest_float('strength', 0.2, 0.7)
    guidance = trial.suggest_float("guidance", 2, 50)

    num_per_trials = 50

    images = []
    while num_per_trials>5: 
      num_per_trials -= 5
      images.extend(pipe(prompt=prompt,negative_prompt=negative_prompt, init_image=init_image, strength=strength, guidance_scale=guidance, num_images_per_prompt = 5).images)
    
    images.extend(pipe(prompt=prompt,negative_prompt=negative_prompt, init_image=init_image, strength=strength, guidance_scale=guidance, num_images_per_prompt = num_per_trials).images)

    scores = []
    for i in range(len(images)):
      num_im = len(os.listdir(p))
      im_name = str(num_im)+"_whole_car_"+str(round(strength,2))+"_"+str(round(guidance,2))+".png"
      whole_im_path = p+im_name
      images[i].save(whole_im_path)
      im_score = get_score(whole_im_path,damaged_im_path)
      
      # name = str(num_im)+", mean_"+str(round(im_score[0],2))+", ssim_"+str(round(im_score[1],2))+", car_"+str(round(im_score[2],2))
      # print(name)
      # plt.figure(figsize = (3,3))
      # plt.title(name)
      # plt.imshow(images[0])
      # plt.show()
      os.remove(whole_im_path)
      if im_score == None: continue
      scores.append(im_score[0])

      # with open(p+"labels/"+im_name.replace("png","txt"), "w") as file:
      #     file.writelines([str(im_score)])
    return sum(scores)/len(scores)
    
    

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", sampler = optuna.samplers.TPESampler(seed=4,n_startup_trials=20),pruner = optuna.pruners.HyperbandPruner())
    study.optimize(objective, n_trials=75)
    print(study.best_trial)
    
#best parameter combination
study.best_params

#score achieved with best parameter combination
study.best_value

fig = plot_contour(study)
print(study)
print(study.trials)
import matplotlib.pyplot as plt
try: fig.savefig('plot_contour.png') 
except: plt.savefig('plot_contour.png')