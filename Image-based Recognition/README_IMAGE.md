
 - **Environment**

This source code was tested in the following environment:

    Python = 3.7.9
    PyTorch = 1.7.0
    torchvision = 0.4.2
    Ubuntu 18.04.5 LTS
    timm = 0.1.26
    cuda = 10.2

 - **Dataset**

(1) Download the `AUCD2` or `SFD3` and organize the structure as follows:

    dataset
    ├── train
    │   ├── c0
    |   |      ├── 1.jpg
    |   |      ├── 2.jpg
    |   |      └── ...
    │   ├── c1
    |   |      ├── 1.jpg
    |   |      ├── 2.jpg
    |   |      └── ...
    │   └── ...
    └── test
        ├── c0
        |      ├── 1.jpg
        |      ├── 2.jpg
        |      └── ...
        ├── c1
        |      ├── 1.jpg
        |      ├── 2.jpg
        |      └── ...
        └── ...

(2) modify the path to the "`train`" and "`test`" folders.

 - **Dependencies**

(1) Imgaug (https://github.com/aleju/imgaug)

    conda install imgaug

(2) timm  
Download the folder "`timm`" from https://github.com/rwightman/pytorch-image-models and save it as:

    source_code
    ├── train_driver_finetune.py
    ├── train_driver_search_OE.py
    ├── train_driver_search.py
    ├── ...
    ├── timm

Note that: we tested the codes on timm = 0.1.26

(3) Pyramidal Convolution

Download the folders "`div`" and "`models`" from https://github.com/iduta/pyconv and save it as:

    source_code
    ├── train_driver_finetune.py
    ├── train_driver_search_OE.py
    ├── train_driver_search.py
    ├── ...
    ├── div
    ├── models

(4) 

Download the "`kd_losses`" folder from https://github.com/AberHu/Knowledge-Distillation-Zoo and save it as:

    source_code
    ├── train_driver_finetune.py
    ├── train_driver_search_OE.py
    ├── train_driver_search.py
    ├── ...
    ├── kd_losses

Note that: if you come across the error "ModuleNotFoundError: No module named 'kd_losses2.kdsvd'", you can simply solve this error by deleting the code "from .kdsvd import KDSVD" in line 10 of "kd_losses/__init__.py".

 - **Training**

This package contains the source code to train the teacher network, search the architecture of the student network, transfer the knowledge of the teacher network to the student network, and finetune the student network. In the following commands, "OE" means use the SKResNeXt50 backbone only as an extractor. That is, for the teacher network, "OE" does not finetune the pertained weights in the SKResNeXt50 backbone, but only update the parameters in the newly added convolutional layers, fully-connected layers, batch-normalization layers, etc. for progressive learning. In our practice, "OE" has a very slightly better performance on the AUCD2.


Step 1: Train the teacher network.

    python train_driver_teacher_OE.py

or

    python train_driver_teacher.py


Step 2: Define the architecture of the student network.

    python train_driver_search_OE.py

or

    python train_driver_search.py


Step 3: Transfer the knowledge of the teacher network to the student network.

    python train_driver_transfer_OE.py --kd_mode=logits

or

    python train_driver_transfer.py --kd_mode=logits


Step 4: Fine-tune the student network.

    python train_driver_finetune.py --OE

or
 

    python train_driver_finetune.py
