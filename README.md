# Moss
Code for Paper ["Memory Oriented Transfer Learning for Semi-Supervised Image Deraining"](https://openaccess.thecvf.com/content/CVPR2021/html/Huang_Memory_Oriented_Transfer_Learning_for_Semi-Supervised_Image_Deraining_CVPR_2021_paper.html)

## Prerequisites
* Python 3
* PyTorch

## Models

We provide the model trained on DDN-SIRR dataset in the following links:

* [Google Driver](https://drive.google.com/drive/folders/1Ob4ATRd5bKtGLEzPhL84saY3tUW33Lu4?usp=sharing) 
* [Jianguo Yun](https://www.jianguoyun.com/p/Deq2B2gQiaCuBxjy9IUE)

## Run

	 CUDA_VISIBLE_DEVICES=0 python test.py --config_file='test.yaml'

Adjust the parameters in 'test.yaml' according to your own settings.
Please goto the 'train' folder for the training code.

## Citation

If you use our codes, please cite the following paper:

	 @inproceedings{huang2021memory,
	   title={Memory Oriented Transfer Learning for Semi-Supervised Image Deraining},
	   author={Huang, Huaibo and Yu, Aijing and He, Ran},
	   booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
	   pages={7732--7741},
	   year={2021},
	  }
 
**The released codes are only allowed for non-commercial use.**
