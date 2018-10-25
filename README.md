# Faster_RCNN_OCR
FCN Faster_RCNN + OCR
# Notice
This PyTorch implementation currently supports text detection by Faster-RCNN and OCR by CRNN in two seperate models.

## Installation
Please refer to [ruotianluo's Faster-RCNN implementation](https://github.com/ruotianluo/pytorch-faster-rcnn#installation), and [meijieru's CRNN implementation](https://github.com/meijieru/crnn.pytorch). [Baidu's Warp-ctc](https://github.com/baidu-research/warp-ctc) is also relied for the translation.

## Prerequisites
  - A basic pytorch installation. The code follows **0.4**. 
  - Python packages you might not have: `cffi`, `opencv-python`, `easydict` (similar to [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)). For `easydict` make sure you have the right version. Xinlei uses 1.6.
  - [tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch) to visualize the training and validation curve. Please build from source to use the latest tensorflow-tensorboard.
  
## Setup data
Please follow the instructions of py-faster-rcnn [here](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to setup VOC and COCO datasets (Part of COCO is done). The steps involve downloading data and optionally creating soft links in the ``data`` folder. Since faster RCNN does not rely on pre-computed proposals, it is safe to ignore the steps that setup proposals.

full_text dataset is customized to have an additional ground truth dict "gt_text" similar to bounding boxes, you can follow the same metrics of VOC to customize your data.

## Performance
TBA

