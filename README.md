# TripoSG: High-Fidelity 3D Shape Synthesis using Large-Scale Rectified Flow Models

<div align="center">

[![Project Page](https://img.shields.io/badge/🏠-Project%20Page-blue.svg)](https://yg256li.github.io/TripoSG-Page/)
[![Paper](https://img.shields.io/badge/📑-Paper-green.svg)](https://arxiv.org/abs/2502.06608)
[![Model](https://img.shields.io/badge/🤗-Model-yellow.svg)](https://huggingface.co/VAST-AI/TripoSG)
[![Online Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/VAST-AI/TripoSG)

**By [Tripo](https://www.tripo3d.ai)**

</div>

![teaser](assets/doc/triposg_teaser.png)

TripoSG is an advanced high-fidelity, high-quality and high-generalizability image-to-3D generation foundation model. It leverages large-scale rectified flow transformers, hybrid supervised training, and a high-quality dataset to achieve state-of-the-art performance in 3D shape generation.

## ✨ Key Features

- **High-Fidelity Generation**: Produces meshes with sharp geometric features, fine surface details, and complex structures
- **Semantic Consistency**: Generated shapes accurately reflect input image semantics and appearance
- **Strong Generalization**: Handles diverse input styles including photorealistic images, cartoons, and sketches
- **Robust Performance**: Creates coherent shapes even for challenging inputs with complex topology

## 🔬 Technical Highlights

- **Large-Scale Rectified Flow Transformer**: Combines RF's linear trajectory modeling with transformer architecture for stable, efficient training
- **Advanced VAE Architecture**: Uses Signed Distance Functions (SDFs) with hybrid supervision combining SDF loss, surface normal guidance, and eikonal loss
- **High-Quality Dataset**: Trained on 2 million meticulously curated Image-SDF pairs, ensuring superior output quality
- **Efficient Scaling**: Implements architecture optimizations for high performance even at smaller model scales

## 🔥 Updates

* [2025-03] Release of TripoSG 1.5B parameter rectified flow model and VAE trained on 2048 latent tokens, along with inference code and interactive demo

## 🔨 Installation

Clone the repo:
```bash
git clone https://github.com/VAST-AI-Research/TripoSG.git
cd TripoSG
```

Create a conda environment (optional):
```bash
conda create -n tripoSG python=3.10
conda activate tripoSG
```

Install dependencies:
```bash
# pytorch (select correct CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/{your-cuda-version}

# other dependencies
pip install -r requirements.txt
```

## 💡 Quick Start

Generate a 3D mesh from an image:
```bash
python -m scripts.inference_triposg --image-input assets/example_data/hjswed.png
```

The required model weights will be automatically downloaded:
- TripoSG model from [VAST-AI/TripoSG](https://huggingface.co/VAST-AI/TripoSG) → `pretrained_weights/TripoSG`
- RMBG model from [briaai/RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4) → `pretrained_weights/RMBG-1.4`

## 💻 System Requirements

- CUDA-enabled GPU with at least 8GB VRAM

## 📝 Tips

- If you want to use the full VAE module (including the encoder part), you need to uncomment the Line-15 in `triposg/models/autoencoders/autoencoder_kl_triposg.py` and install `torch-cluster`.

## 🤝 Community & Support

- **Interactive Demo**: Try TripoSG on [Hugging Face Spaces](https://huggingface.co/spaces/VAST-AI/TripoSG)
- **Issues & Discussions**: Use GitHub Issues for bug reports and feature requests
- **Contributing**: We welcome contributions!

## 📚 Citation

```
@article{li2025triposg,
  title={TripoSG: High-Fidelity 3D Shape Synthesis using Large-Scale Rectified Flow Models},
  author={Li, Yangguang and Zou, Zi-Xin and Liu, Zexiang and Wang, Dehu and Liang, Yuan and Yu, Zhipeng and Liu, Xingchao and Guo, Yuan-Chen and Liang, Ding and Ouyang, Wanli and others},
  journal={arXiv preprint arXiv:2502.06608},
  year={2025}
}
```

## ⭐ Acknowledgements

We would like to thank the following open-source projects and research works that made TripoSG possible:

- [DINOv2](https://github.com/facebookresearch/dinov2) for their powerful visual features
- [RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4) for background removal
- [🤗 Diffusers](https://github.com/huggingface/diffusers) for their excellent diffusion model framework
- [HunyuanDiT](https://github.com/Tencent/HunyuanDiT) for DiT
- [3DShape2VecSet](https://github.com/1zb/3DShape2VecSet) for 3D shape representation

We are grateful to the broader research community for their open exploration and contributions to the field of 3D generation.
