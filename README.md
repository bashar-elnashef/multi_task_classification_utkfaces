# Multi-task Classification UTKFace Pytorch Project 

In this project a multi-task classification on the UTKFace data is implemented over the three tasks, age, gender and race classification. 

### [UTKFace dataset](https://susanqq.github.io/UTKFace/)

UTKFace dataset is a large-scale face dataset with long age span (range from 0 to 116 years old). The dataset consists of over 20,000 face images with annotations of age, gender, and ethnicity. The images cover large variation in pose, facial expression, illumination, occlusion, resolution, etc. This dataset could be used on a variety of tasks, e.g., face detection, age estimation, age progression/regression, landmark localization, etc.

**Note**: Although the dataset can be downloaded form the official website [here](https://susanqq.github.io/UTKFace/), I have uploaded a copy of the "Aligned&Cropped Faces" to my personal google drive to make things easier. <p>

**Note**: I have also added a standalone demo_notebook file for ease of use and proof of concept. The optimized version of this are incorperated into the project.   

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

* [Project]()
	* [Requirements](#requirements)
	* [Usage](#usage)
		* [Using config files](#using-config-files)
		* [Resuming from checkpoints](#resuming-from-checkpoints)
	* [TODOs](#todos)
	* [Acknowledgements](#acknowledgements)

<!-- /code_chunk_output -->

## Requirements
* Python >= 3.5 (3.6 recommended)
* PyTorch >= 0.4 (1.2 recommended)
* tqdm (Optional for `test.py`)
* tensorboard >= 1.14 (see [Tensorboard Visualization](#tensorboard-visualization))

## Usage
The code in this repo excuted from the command line. 
Try `python train.py -c config.json` to run code.

### Using config files
Modify the configurations in `.json` config files, then run:

  ```
  python train.py --config config.json
  ```

### Resuming from checkpoints
You can resume from a previously saved checkpoint by:

  ```
  python train.py --resume path/to/checkpoint
  ```

### Testing
You can test trained model by running `test.py` passing path to the trained checkpoint by `--resume` argument.

### Validation data
To split validation data from a data loader, call `BaseDataLoader.split_validation()`, then it will return a data loader for validation of size specified in your config file.
The `validation_split` can be a ratio of validation set per total data(0.0 <= float < 1.0), or the number of samples (0 <= int < `n_total_samples`).

**Note**: the `split_validation()` method will modify the original data loader
**Note**: `split_validation()` will return `None` if `"validation_split"` is set to `0`


### Tensorboard Visualization
This template supports Tensorboard visualization by using either  `torch.utils.tensorboard` or [TensorboardX](https://github.com/lanpa/tensorboardX).

1. **Install**

    If you are using pytorch 1.1 or higher, install tensorboard by 'pip install tensorboard>=1.14.0'.

    Otherwise, you should install tensorboardx. Follow installation guide in [TensorboardX](https://github.com/lanpa/tensorboardX).

2. **Run training** 

    Make sure that `tensorboard` option in the config file is turned on.

    ```
     "tensorboard" : true
    ```

3. **Open Tensorboard server** 

    Type `tensorboard --logdir saved/log/` at the project root, then server will open at `http://localhost:6006`


## TODOs
- [ ] Data Exploration and Balancing 
- [ ] Include other optimizers
- [ ] Research multi-task traning paradigms
- [ ] Explore and implement additional multi-class architectures
  - [ ] Deep Relationship Networks (2015)   
  - [ ] Fully Adapted Feature Sharing (2016) 
  - [ ] Cross Stitch Networks (2017) 
  - [ ] Bayesian Networks (2017)
  - [ ] Sluice Networks (2017)
  - [ ] Cross Correlated Networks (2018) 
  - [X] [HydraNets (2018)](https://openaccess.thecvf.com/content_cvpr_2018/papers/Mullapudi_HydraNets_Specialized_Dynamic_CVPR_2018_paper.pdf)
  - [ ] Google Pathways (2021) 

<!-- ## License
This project is licensed under the MIT License. See  LICENSE for more details -->

## Acknowledgements
This project is build on the [pytorch-template](https://github.com/victoresque/pytorch-template) by [Victor Huang
](https://github.com/victoresque)
