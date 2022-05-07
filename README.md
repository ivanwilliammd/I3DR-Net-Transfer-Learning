# I3DR-Net Transfer Learning
*Official implementation code for "Lung Nodule Detection and Classification from Thorax CT-Scan Using RetinaNet with Transfer Learning" and "Lung Nodule Texture Detection and Classification Using 3D CNN."*

Contributor:
[@ivanwilliammd](https://github.com/ivanwilliammd), [@wawancenggoro](https://github.com/wawancenggoro), [@sliawatimena](https://github.com/sliawatimena)

## Related publication:

1. [William Harsono I, Liawatimena S, Wawan Cenggoro T. "Lung Nodule Detection and Classification from Thorax CT-Scan Using RetinaNet with Transfer Learning". Journal of King Saud University - Computer and Information Sciences. 2020.](https://doi.org/10.1016/j.jksuci.2020.03.013) 
2. [Ivan William Harsono, Suryadiputra Liawatimena, Tjeng Wawan Cenggoro. "Lung Nodule Texture Detection and Classification Using 3D CNN." CommIT Journal 13.2, 2019.](https://journal.binus.ac.id/index.php/commit/article/view/5995)
3. [Lin, Tsung-Yi, et al. "Focal Loss for Dense Object Detection" TPAMI, 2018.](https://arxiv.org/abs/1708.02002)
4. [Carreira, Joao, and Andrew Zisserman. "Quo vadis, action recognition? a new model and the kinetics dataset." proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017.](https://arxiv.org/abs/1705.07750)
5. [Jaeger, Paul et al. "Retina U-Net: Embarrassingly Simple Exploitation of Segmentation Supervision for Medical Object Detection" , 2018](https://arxiv.org/abs/1811.08661)

## How to cite this code
*Please cite our paper [1] & [2]*

## Adaptation
This codes are adapted from medicaldetectiontoolkit from pfjaeger for [Jaeger et al, Medical Detection Toolkit](https://github.com/pfjaeger/medicaldetectiontoolkit) [3, 5] and [Kinetics I3D](https://github.com/hassony2/kinetics_i3d_pytorch) (I3D Backbone) [4] for malignancy (malignant for score>3, benign otherwise), texture classification (groundglass, subsolid, and solid), and nodule detection (Unsupressed RoI will be scored as 1 (foreground), while supressed RoI will be scored as 0 (background))

The LIDC needs to be preprocessed using [DICOM preprocessing scripts](https://github.com/ivanwilliammd/DICOM-data-preprocessing-script) and [Batch generators](https://github.com/ivanwilliammd/batchgenerators) for training


## This repository support experiments (experiments/ folder):
1. lidc_exp_malignancy: preprocessing and data loader for LIDC malignancy clasification (2 class + 1 bg class)
2. lidc_exp_texture: modified preprocessing and data loader for LIDC texture clasification (3 class + 1 bg class)
3. moscow_exp_texture: modified preprocessing and data loader for private dataset texture clasification  (3 class + 1 bg class)
4. moscow_exp_subtlety: modified preprocessing and data loader for private dataset subtlety clasification  (2 class + 1 bg class) --> 2 fg class : obvious (solid); subtle (subsolid, groundglass)
5. lidc_exp_nodule: modified preprocessing and data loader for LIDC nodule clasification (1 fg+ 1 bg class)
6. moscow_exp_nodule: modified preprocessing and data loader for Moscow nodule clasification (1 fg+ 1 bg class)


## Install package and dependency

> Setup package in virtual environment
```
git clone https://github.com/ivanwilliammd/I3DR-Net-Transfer-Learning
cd I3DR-Net-Transfer-Learning
pip3 install -e .
```

> Install MIC-DKFZ batch-generators 
```
git clone https://github.com/ivanwilliammd/batchgenerators
cd batchgenerators
pip3 install -e .
```

## Install NMS, RoI, Align for CUDA 9.x and pytorch 0.4.1 (Pre-compiled for Tesla P100.)

Non-Maximum Suppression 
*taken from [pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn) and added adaption for 3D*
```
cd cuda_functions/nms_2D/src/cuda/
nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_60
cd ../../
python build.py
cd ../../

cd cuda_functions/nms_3D/src/cuda/
nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_60
cd ../../
python build.py
cd ../../
```
RoiAlign 
*taken from [RoiAlign](https://github.com/longcw/RoIAlign.pytorch), fixed according to [this bug report](https://hackernoon.com/how-tensorflows-tf-image-resize-stole-60-days-of-my-life-aba5eb093f35), and added adaption for 3D*

```
cd cuda_functions/roi_align_2D/roi_align/src/cuda/
nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_60
cd ../../
python build.py
cd ../../../

cd cuda_functions/roi_align_3D/roi_align/src/cuda/
nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_60
cd ../../
python build.py
cd ../../../
```

| GPU | arch |
| --- | --- |
| TitanX | sm_52 |
| GTX 960M | sm_50 |
| Tesla P100 | sm_60 |
| Tesla P4 | sm_61 |
| GTX 10XX series | sm_61 |
| GTX 1080 (Ti) | sm_61 |
| Tesla V100 | sm_70 |

*for more compute capability information, please see: (https://en.wikipedia.org/wiki/CUDA )*

## Execute
1. Set I/O paths, model and training specifics in the configs file: i3dr-net/experiments/your_experiment/configs.py
2. Train the model: 

	```
	python exec.py --mode train --exp_source experiments/my_experiment --exp_dir path/to/experiment/directory       
	```

	This copies snapshots and monitoring diagram of configs and model to the specified exp_dir, where all outputs will be saved. By default, the data is split into 60% training and 20% validation and 20% testing data to perform a 5-fold cross validation (can be changed to hold-out test set in configs) and all folds will be trained iteratively. In order to train a single fold, specify it using the folds arg: 

	```
	python exec.py --folds 0 1 2 .... # specify any combination of folds [0-4]
	```

3. Run inference:
	
	```
	python exec.py --mode test --exp_dir path/to/experiment/directory 
	```
	
	This runs the prediction pipeline and saves all results to exp_dir.

4. Add pretrained ImageNet I3D RGB Weights:
	
	```
	python exec.py --mode train --exp_source experiments/my_experiment --exp_dir path/to/experiment/directory --rgb_weights_path weight/model_rgb.pth     
	``` 
	
	This runs the prediction pipeline and saves all results to exp_dir.
	

## Example of Execution Code
```
python exec.py --mode train_test --folds 0 --exp_source experiments/lidc_exp_subtlety/ --exp_dir ~/I3DR-Models/LIDC-Solidity_TransferI3D --rgb_weights_path weight/model_rgb.pth
```
```
python exec.py --mode train_test --folds 0 --exp_source experiments/moscow_exp_subtlety/ --exp_dir ~/I3DR-Models/Moscow-Solidity_TransferI3D --rgb_weights_path weight/model_rgb.pth
```

*Last updated July 2nd 2019*
