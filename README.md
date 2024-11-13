- 👋 Hi, I’m @bai1850
- 
# First clone the repo, and use the commands in the repo

import torch
import requests
from PIL import Image
import numpy as np
from torchvision.utils import make_grid, save_image
from diffusers import DiffusionPipeline  # only tested on diffusers[torch]==0.19.3, may have conflicts with newer versions of diffusers

def load_wonder3d_pipeline():

    pipeline = DiffusionPipeline.from_pretrained(
    'flamehaze1115/wonder3d-v1.0', # or use local checkpoint './ckpts'
    custom_pipeline='flamehaze1115/wonder3d-pipeline',
    torch_dtype=torch.float16
    )

    # enable xformers
    pipeline.unet.enable_xformers_memory_efficient_attention()

    if torch.cuda.is_available():
        pipeline.to('cuda:0')
    return pipeline

pipeline = load_wonder3d_pipeline()

# Download an example image.
cond = Image.open(requests.get("https://d.skis.ltd/nrp/sample-data/lysol.png", stream=True).raw)

# The object should be located in the center and resized to 80% of image height.
cond = Image.fromarray(np.array(cond)[:, :, :3])

# Run the pipeline!
images = pipeline(cond, num_inference_steps=20, output_type='pt', guidance_scale=1.0).images

result = make_grid(images, nrow=6, ncol=2, padding=0, value_range=(0, 1))

save_image(result, 'result.png')👀 I’m interested in ...
- 🌱 I’m currently learning ...
- 💞️ I’m looking to collaborate on ...
- 📫 How to reach me ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...

<!---
bai1850/bai1850 is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
