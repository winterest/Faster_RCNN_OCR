# Faster_RCNN_OCR
FCN Faster_RCNN + OCR
# Notice
**10/25/18** This PyTorch implementation currently supports text detection by Faster-RCNN and OCR by CRNN in two seperate models.

## Installation
Please refer to [ruotianluo's Faster-RCNN implementation](https://github.com/ruotianluo/pytorch-faster-rcnn#installation), and [meijieru's CRNN implementation](https://github.com/meijieru/crnn.pytorch). [Baidu's Warp-ctc](https://github.com/baidu-research/warp-ctc) is also relied for the translation.

## Prerequisites
  - A basic pytorch installation. The code follows **0.4**. 
  - Python packages you might not have: `cffi`, `opencv-python`, `easydict` (similar to [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)). For `easydict` make sure you have the right version. Xinlei uses 1.6.
  - [tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch) to visualize the training and validation curve. Please build from source to use the latest tensorflow-tensorboard.
  
## Setup data
Please follow the instructions of py-faster-rcnn [here](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to setup VOC and COCO datasets (Part of COCO is done). The steps involve downloading data and optionally creating soft links in the ``data`` folder. Since faster RCNN does not rely on pre-computed proposals, it is safe to ignore the steps that setup proposals.

full_text dataset is customized to have an additional ground truth dict "gt_text" similar to bounding boxes, you can follow the same metrics of VOC to customize your data.

## Training
1. Download pre-trained models and weights. The current code support VGG16 and Resnet V1 models. Pre-trained models are provided by [pytorch-vgg](https://github.com/jcjohnson/pytorch-vgg.git) and [pytorch-resnet](https://github.com/ruotianluo/pytorch-resnet) (the ones with caffe in the name), you can download the pre-trained models and set them in the ``data/imagenet_weights`` folder. For example for VGG16 model, you can set up like:
   ```Shell
   mkdir -p data/imagenet_weights
   cd data/imagenet_weights
   python # open python in terminal and run the following Python code
   ```
   ```Python
   import torch
   from torch.utils.model_zoo import load_url
   from torchvision import models

   sd = load_url("https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg16-00b39a1b.pth")
   sd['classifier.0.weight'] = sd['classifier.1.weight']
   sd['classifier.0.bias'] = sd['classifier.1.bias']
   del sd['classifier.1.weight']
   del sd['classifier.1.bias']

   sd['classifier.3.weight'] = sd['classifier.4.weight']
   sd['classifier.3.bias'] = sd['classifier.4.bias']
   del sd['classifier.4.weight']
   del sd['classifier.4.bias']

   torch.save(sd, "vgg16.pth")
   ```
   ```Shell
   cd ../..
   ```
   For Resnet101, you can set up like:
   ```Shell
   mkdir -p data/imagenet_weights
   cd data/imagenet_weights
   # download from my gdrive (link in pytorch-resnet)
   mv resnet101-caffe.pth res101.pth
   cd ../..
   ```

   For Mobilenet V1, you can set up like:
   ```Shell
   mkdir -p data/imagenet_weights
   cd data/imagenet_weights
   # download from my gdrive (https://drive.google.com/open?id=0B7fNdx_jAqhtZGJvZlpVeDhUN1k)
   mv mobilenet_v1_1.0_224.pth.pth mobile.pth
   cd ../..
   ```

2. Train (and test, evaluation)
  ```Shell
  ./experiments/scripts/train_faster_rcnn.sh [GPU_ID] [DATASET] [NET]
  # GPU_ID is the GPU you want to test on
  # NET in {vgg16, res50, res101, res152} is the network arch to use
  # DATASET {pascal_voc, pascal_voc_0712, coco} is defined in train_faster_rcnn.sh
  # Examples:
  ./experiments/scripts/train_faster_rcnn.sh 0 pascal_voc vgg16
  ./experiments/scripts/train_faster_rcnn.sh 1 coco res101
  ```
## Performance
TBA

