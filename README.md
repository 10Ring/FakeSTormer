# [ICCV2025] [FakeSTormer] Vulnerability-Aware Spatio-Temporal Learning for Generalizable Deepfake Video Detection

![alt text](./demo/method.png?raw=true)
This is an official implementation of FakeSTormer! [[📜Paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Nguyen_Vulnerability-Aware_Spatio-Temporal_Learning_for_Generalizable_Deepfake_Video_Detection_ICCV_2025_paper.pdf)]


## Updates
- [ ] *-/-/- : Code refactor 🚀.*
- [x] 26/11/2025:*Official release of code (v1) and pretrained weights 🌈.*
- [x] 08/07/2025: *First version pre-released for this open source code 🌱.*
- [x] 26/06/2025: *FakeSTormer has been accepted to ICCV2025 🎉.*


## Abstract
Detecting deepfake videos is highly challenging given the complexity of characterizing spatio-temporal artifacts. Most existing methods rely on binary classifiers trained using real and fake image sequences, therefore hindering their generalization capabilities to unseen generation methods. Moreover, with the constant progress in generative Artificial Intelligence (AI), deepfake artifacts are becoming imperceptible at both the spatial and the temporal levels, making them extremely difficult to capture.  To address these issues, we propose a fine-grained deepfake video detection approach called FakeSTormer that enforces the modeling of subtle spatio-temporal inconsistencies while avoiding overfitting. Specifically, we introduce a multi-task learning framework that incorporates two auxiliary branches for explicitly attending artifact-prone spatial and temporal regions. Additionally, we propose a video-level data synthesis strategy that generates pseudo-fake videos with subtle spatio-temporal artifacts, providing high-quality samples and hand-free annotations for our additional branches. Extensive experiments on several challenging benchmarks demonstrate the superiority of our approach compared to recent state-of-the-art methods.


## Main Results
Results on 6 datasets ([CDF2](https://github.com/yuezunli/celeb-deepfakeforensics), [DFW](https://github.com/deepfakeinthewild/deepfake-in-the-wild), [DFD](https://blog.research.google/2019/09/contributing-data-to-deepfake-detection.html), [DFDC, DFDCP](https://ai.meta.com/datasets/dfdc/), and [DiffSwap](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_DiffSwap_High-Fidelity_and_Controllable_Face_Swapping_via_3D-Aware_Masked_Diffusion_CVPR_2023_paper.pdf)) under cross-dataset evaluation setting reported by AUC (%) at video-level.

|  |  CDF2  |    DFW     |     DFD    |     DFDC   |   DFDCP | DiffSwap |
|--|--------|------------|------------|------------|---------|-----------|
|<table><thead><tr><th>Compression</th></tr></thead><tbody><tr><td>c23</td></tr></tbody><tbody><tr><td>c0</td></tr></tbody></table>|<table><thead><tr><th>AUC</th></tr></thead><tbody><tr><td>92.4</td></tr></tbody><tbody><tr><td>96.5</td></tr></tbody></table>|<table><thead><tr><th>AUC</th></tr></thead><tbody><tr><td>74.2</td></tr></tbody><tbody><tr><td>76.3</td></tr></tbody></table>|<table><thead><tr><th>AUC</th></tr></thead><tbody><tr><td>98.5</td></tr></tbody><tbody><tr><td>98.9</td></tr></tbody></table>|<table><thead><tr><th>AUC</th></tr></thead><tbody><tr><td>74.6</td></tr></tbody><tbody><tr><td>77.6</td></tr></tbody></table>|<table><thead><tr><th>AUC</th></tr></thead><tbody><tr><td>90.0</td></tr></tbody><tbody><tr><td>94.1</td></tr></tbody></table>|<table><thead><tr><th>AUC</th></tr></thead><tbody><tr><td>96.9</td></tr></tbody><tbody><tr><td>97.7</td></tr></tbody></table>


## Recommended Environment
*For experiment purposes, we encourage the installment of the following libraries. Both Conda or Python virtual env should work.*

* CUDA: 11.4
* [Python](https://www.python.org/): >= 3.8.x
* [PyTorch](https://pytorch.org/get-started/previous-versions/): 1.8.0
* [TensorboardX](https://github.com/lanpa/tensorboardX): 2.5.1
* [ImgAug](https://github.com/aleju/imgaug): 0.4.0
* [Scikit-image](https://scikit-image.org/): 0.17.2
* [Torchvision](https://pytorch.org/vision/stable/index.html): 0.9.0
* [Albumentations](https://albumentations.ai/): 1.1.0
* [mmcv](https://github.com/open-mmlab/mmcv): 1.6.1
* [natsort](https://pypi.org/project/natsort/): 8.4.0



## Pre-trained Models
* 📌 *The pre-trained weights of FakeSTormer can be found [here](https://www.dropbox.com/scl/fo/elk2szqf0du4l6zm5job9/AAdVmNH--6ywHBZGNQJlR5o?rlkey=5kde7vj4wklrx1jwdul0m6g46&st=czw4szw0&dl=0)*


## Docker Build (Optional)
*We further provide an optional Docker file that can be used to build a working env with Docker. More detailed steps can be found [here](dockerfiles/README.md).*

1.  Install docker to the system (skip the step if docker has already been installed):
    ```shell
    sudo apt install docker
    ```
2. To start your docker environment, please go to the folder **dockerfiles**:
   ```shell
   cd dockerfiles
   ```
3. Create a docker image (you can put any name you want):
    ```shell
    docker build --tag 'fakestormer' .
    ```


## Quickstart
1. **Preparation**

    1. ***Prepare environment***

        Installing main packages as the recommended environment. *Note that we recommend building mmcv from source as below.*
        > git clone https://github.com/open-mmlab/mmcv.git \
        cd mmcv \
        git checkout v1.6.1 \
        MMCV_WITH_OPS=1 pip install -e .
    
    2. ***Prepare dataset***
        
        1. Downloading [FF++](https://github.com/ondyari/FaceForensics) *Original* dataset for training data preparation. Following the original split convention, it is firstly used to randomly extract frames and facial crops:
            ```
            python package_utils/images_crop.py -d {dataset} \
            -c {compression} \
            -n {num_frames} \
            -t {task}
            ```
            (*This script can also be utilized for cropping faces in other datasets such as [CDF2](https://github.com/yuezunli/celeb-deepfakeforensics), [DFD](https://blog.research.google/2019/09/contributing-data-to-deepfake-detection.html), [DFDCP, DFDC](https://ai.meta.com/datasets/dfdc/) for cross-evaluation test. You do not need to run crop for [DFW](https://github.com/deepfakeinthewild/deepfake-in-the-wild) as the data is already preprocessed*).
            
            | Parameter | Value | Definition  |
            | --- | --- | --- |
            | -d | Subfolder in each dataset. For example: *['Face2Face','Deepfakes','FaceSwap','NeuralTextures', ...]*| You can use one of those datasets.|
            | -c | *['raw','c23','c40']*| You can use one of those compressions|
            | -n | *256*  | Number of frames (*default* 32 for val/test and 256 for train) |
            | -t | *['train', 'val', 'test']* | Default train|
            
            These faces cropped are saved for online pseudo-fake generation in the training process, following the data structure below:
            
            ```
            ROOT = '/data/deepfake_cluster/datasets_df'
            └── Celeb-DFv2
                └──...
            └── FF++
                └── c0
                └── c23
                    ├── test
                    │   └── videos
                    │       └── Deepfakes
                    |           ├── 000_003
                    |           ├── 044_945
                    |           ├── 138_142
                    |           ├── ...
                    │       ├── Face2Face
                    │       ├── FaceSwap
                    │       ├── NeuralTextures
                    │       └── original
                    |   └── frames
                    ├── train
                    │   └── videos
                    │       └── aligned
                    |           ├── 001
                    |           ├── 002
                    |           ├── ...  
                    │       └── original
                    |           ├── 001
                    |           ├── 002
                    |           ├── ...
                    |   └── frames
                    └── val
                        └── videos
                            ├── aligned
                            └── original
                        └── frames
                └── c40
            ```
        
        2. Downloading **Dlib** [[81]](https://github.com/codeniko/shape_predictor_81_face_landmarks) facial landmarks detector pretrained and place into ```/pretrained/``` for *SBI* synthesis.

        3. Landmarks detection. After completing the following script running, a file that stores metadata information of the data is saved at ```processed_data/c23/{SPLIT}_FaceForensics_videos_<n_landmarks>.json```.
            ```
            python package_utils/geo_landmarks_extraction.py \
            --config configs/data_preprocessing_c23.yaml \
            --extract_landmarks
            ```
        
2. **Training script**

    We offer a number of config files for different compression levels of training data. For *c23*, opening ```configs/temporal/FakeSTormer_base_c23.yaml```, please make sure you set ```TRAIN: True``` and ```FROM_FILE: True``` and run:
    ```
    .scripts/fakestormer_sbi.sh
    ```

    Otherwise, with *[c0, c40]*, the config file is ```configs/temporal/FakeSTormer_base_[c0, c40].yaml```. You can also find other configs for other network architectures in the ```configs/``` folder. 


3. **Testing script**

    Opening ```configs/temporal/FakeSTormer_base_c23.yaml```, with ```subtask: eval``` in the *test* section, we support evaluation mode, please turn off ```TRAIN: False``` and ```FROM_FILE: False``` and run:
    ```
    .scripts/test_fakestormer.sh
    ```
    For others (.e.g., data compression levels, network architectures), please change the path of the coressponding config file.

    > ⚠️ *Please make sure you set the correct path to your download pre-trained weights in the config files.*

    > ℹ️ *Flip test can be used by setting ```flip_test: True```*
    
    > ℹ️ *The mode for single video inference is also provided, please set ```sub_task: test_vid``` and pass a video path as an argument in test.py*


## Contact
Please contact dat.nguyen@uni.lu. Any questions or discussions are welcomed!


## License
This software is © University of Luxembourg and is licensed under the snt academic license. See [LICENSE](NOTICE)


## Acknowledge
We acknowledge the excellent implementation from [OpenMMLab](https://github.com/open-mmlab) ([mmengine](https://github.com/open-mmlab/mmengine), [mmcv](https://github.com/open-mmlab/mmcv)), [SBI](https://github.com/mapooon/SelfBlendedImages), and [LAA-Net](https://github.com/10Ring/LAA-Net).


## Citation
Please kindly consider citing our papers in your publications.
```
@inproceedings{nguyen2025vulnerability,
  title={Vulnerability-Aware Spatio-Temporal Learning for Generalizable Deepfake Video Detection},
  author={Nguyen, Dat and Astrid, Marcella and Kacem, Anis and Ghorbel, Enjie and Aouada, Djamila},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={10786--10796},
  year={2025}
}
```
