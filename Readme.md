# StageInteractor: Query-based Object Detector with Cross-stage Interaction

Our paper [StageInteractor: Query-based Object Detector with Cross-stage Interaction](https://openaccess.thecvf.com/content/ICCV2023/papers/Teng_StageInteractor_Query-based_Object_Detector_with_Cross-stage_Interaction_ICCV_2023_paper.pdf) has been accepted by ICCV 2023.

## Installation

Please refer to [get_started.md](docs/get_started.md) for installation.

We also provide the requirements here: 

```bash
conda create -n openmmdet python=3.7
conda activate openmmdet

conda install pytorch==1.10.0 cudatoolkit=11.3 -c pytorch

pip install openmim
mim install mmcv-full==1.3.3

pip install torchvision==0.11.1
pip install setuptools==59.5.0

pip install -e .

```

## Getting Started

Our code is mainly based on: **[AdaMixer](https://github.com/MCG-NJU/AdaMixer)** and **[MMDetection](https://github.com/open-mmlab/mmdetection)**.

Please see [get_started.md](docs/get_started.md) for the basic usage of **[MMDetection](https://github.com/open-mmlab/mmdetection)**.
We provide [colab tutorial](demo/MMDet_Tutorial.ipynb), and full guidance for quick run [with existing dataset](docs/1_exist_data_model.md) and [with new dataset](docs/2_new_data_model.md) for beginners.
There are also tutorials for [finetuning models](docs/tutorials/finetune.md), [adding new dataset](docs/tutorials/new_dataset.md), [designing data pipeline](docs/tutorials/data_pipeline.md), [customizing models](docs/tutorials/customize_models.md), [customizing runtime settings](docs/tutorials/customize_runtime.md) and [useful tools](docs/useful_tools.md).

For frequently asked questions, you can refer to [issues of AdaMixer](https://github.com/MCG-NJU/AdaMixer/issues?q=is%3Aissue+is%3Aclosed) and [FAQ](docs/faq.md).

## Training

Here is an example to run our code with `resnext101_32x4d` as backbones:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=10020 tools/train.py ./configs/stageinteractor/stageinteractor_dx101_300_query_crop_mstrain_480-800_3x_coco.py --launcher pytorch
```

Here is an example to run our code with `Swin-S` as backbones:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=10021 tools/train.py ./configs/stageinteractor/stageinteractor_swin_s_300_query_crop_mstrain_480-800_3x_coco.py --launcher pytorch
```

## Testing

Here is an example to run our code with `resnext101_32x4d` as backbones:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=10025 tools/test.py ./configs/stageinteractor/stageinteractor_dx101_300_query_crop_mstrain_480-800_3x_coco.py ./work_dirs/stageinteractor_dx101_300_query_crop_mstrain_480-800_3x_coco_0725_1348/epoch_36.pth --launcher pytorch --out ./work_dirs/stageinteractor_dx101_300_query_crop_mstrain_480-800_3x_coco_0725_1348/res.pkl  --eval bbox
```

Here is an example to run our code with `Swin-S` as backbones:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=10025 tools/test.py ./configs/stageinteractor/stageinteractor_swin_s_300_query_crop_mstrain_480-800_3x_coco.py ./work_dirs/stageinteractor_swin_s_300_query_crop_mstrain_480-800_3x_coco/epoch_36.pth --launcher pytorch --eval bbox 
```

## Results

Checkpoints and logs are available at [google drive](https://drive.google.com/drive/folders/1o0LehP263Mb96zJ21YkE_E16eivluMiU?usp=sharing).

| config |  detector | backbone  | APval | APtest |
| :------: | :-------: | :-------:  | :-------: | :-------: |
| [config](configs/stageinteractor/stageinteractor_dx101_300_query_crop_mstrain_480-800_3x_coco.py) | StageInteractor (3x schedule, 300 queries)  |  X101-DCN|  51.3  | 51.3  |
| [config](configs/stageinteractor/stageinteractor_swin_s_300_query_crop_mstrain_480-800_3x_coco.py) | StageInteractor (3x schedule, 300 queries)   |  Swin-S  |  52.7  | 52.7   |

## Acknowledgement

Our code is mainly based on: **[AdaMixer](https://github.com/MCG-NJU/AdaMixer)** and **[MMDetection](https://github.com/open-mmlab/mmdetection)**.

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```
@InProceedings{Teng_2023_ICCV,
    author    = {Teng, Yao and Liu, Haisong and Guo, Sheng and Wang, Limin},
    title     = {StageInteractor: Query-based Object Detector with Cross-stage Interaction},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {6577-6588}
}
```
