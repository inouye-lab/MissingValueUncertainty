#!/bin/bash
# Downloads the models used by CoPaint image inpainting for the CelebA dataset.
# These models are not committed, so running this script is necessary if you plan to use the CoPaint generator

(
# download pretrained models
mkdir -p models/checkpoints
cd models/checkpoints

# model pretrained on ImageNet
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt # Trained by OpenAI
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt  # Trained by OpenAI
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/512x512_classifier.pt
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/512x512_diffusion.pt

# model pretrained on CelebA
gdown https://drive.google.com/uc?id=1norNWWGYP3EZ_o05DmoW1ryKuKMmhlCX
)