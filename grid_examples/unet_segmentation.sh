#!/bin/bash
# convert to multi-gpu training by selectiing an instance with multiple gpus from https://docs.grid.ai/products/runs/machines and incrementing the gpus flag
grid run --name unet --instance_type=g4dn.xlarge --gpus=1 ../flash_examples/semantic_segmentation_unet.py

# multi-gpu example
grid run --name unet-multi-gpu --instance_type=g3.8xlarge --gpus=2 ../flash_examples/semantic_segmentation_unet.py

# multi-gpu + HPO example
grid run --name unet-multi-gpu-hpo --instance_type=g3.8xlarge --gpus=2 ../flash_examples/semantic_segmentation_unet.py --max_epochs="[32, 64, 128]"
