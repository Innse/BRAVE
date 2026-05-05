
![header](https://capsule-render.vercel.app/api?type=waving&height=140&color=gradient&text=BRAVE:&section=header&fontAlign=12&fontSize=45&textBg=false&descAlignY=45&fontAlignY=20&descSize=23&desc=A%20Breast%20Vision%20Pathology%20Foundation%20Model%20for%20Real-world%20Clinical%20Utility&descAlign=52)
[![Demo](https://img.shields.io/badge/Demo-Live%20Preview-blue?logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgaGVpZ2h0PSIxNiIgdmlld0JveD0iMCAwIDE2IDE2Ij48Y2lyY2xlIGN4PSI4IiBjeT0iOCIgcj0iNyIgZmlsbD0iI2ZmZmZmZiIvPjxwYXRoIGQ9Ik02IDQuNzV2Ni41bDUtMy4yNS01LTMuMjVaIiBmaWxsPSIjMUE3M0U4Ii8%2BPC9zdmc%2B&logoColor=white)](https://smartpathology.top/login)
[![X (formerly Twitter) Follow](https://img.shields.io/twitter/follow/SMARTLab_HKUST%20)](https://x.com/SMARTLab_HKUST)

> ❗️❗️ Model weights for reviewer evaluation can be accessed via [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/r/personal/yxueb_connect_ust_hk/Documents/project/BRAVE/ckpt/brave.pth?csf=1&web=1&e=e1m5gy). To request a demo trial account or ask about access, please contact [Yingxue Xu](mailto:yxueb@connect.ust.hk).

---

## Table of Contents

- [1. Pretraining](#1-pretraining)
  - [Requirements](#requirements)
  - [Data Preparation](#data-preparation)
  - [Training](#training)
  - [Architecture & Training Mode](#architecture--training-mode)
  - [Output](#output)
- [2. Downstream - Classification](#2-downstream---classification)
  - [Environment Setup](#environment-setup)
  - [Data Preparation](#data-preparation-1)
  - [Training](#training-1)
  - [Output](#output-1)
- [3. Downstream - Survival](#3-downstream---survival)
  - [Environment Setup](#environment-setup-1)
  - [Data Preparation](#data-preparation-2)
  - [Training](#training-2)
  - [Output](#output-2)
- [System Requirements](#system-requirements)
- [Environments Summary](#environments-summary)
- [License and Terms of Tuse](#license-and-terms-of-tuse)

## 1. Pretraining

### Requirements

Create the recommended environment from the repository file:

```bash
conda env create -f pretrain/environment.yml
conda activate brave
```

Dependencies in the repository pretraining environment include:

```text
Python 3.9
torch 2.7.1
torchvision 0.22.1
timm 1.0.19
peft 0.17.0
loralib 0.1.2
h5py 3.14.0
wandb 0.21.0
numpy 1.26.4
Pillow 11.3.0
```

### Data Preparation

#### Patch files (H5 format)

Patches should be stored in `.h5` files. You can use the [PrePATH](https://github.com/birkhoffkiki/PrePATH) repository to prepare these files from Whole Slide Images (WSIs).

Each file must contain a `patches` dataset where each entry is a JPEG/PNG-encoded image byte string:

```
patches/
  0  → raw image bytes (JPEG/PNG)
  1  → raw image bytes
  ...
```

#### Index JSON

Create a JSON file listing all patches to train on. It should be a list of `[image_index, h5_filename]` pairs, where `h5_filename` is a path relative to `--h5_root`:

```json
[
  [0, "slide_001.h5"],
  [1, "slide_001.h5"],
  [0, "slide_002.h5"],
  ...
]
```

### Training

Launch multi-GPU training with `torchrun`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 pretrain/main_dino.py \
    --h5_root /path/to/h5_patches/ \
    --data_path /path/to/pretrain_index.json \
    --output_dir /path/to/output/ \
    --batch_size_per_gpu 32 \
    --local_crops_size 98 \
    --epochs 100 \
    --wandb_proj_name my_project \
    --wandb_exp_name my_experiment
```

A convenience script is provided at `pretrain/run.sh` — edit the path variables at the top before running.

#### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--arch` | `virchow2` | Model backbone. |
| `--mode` | `lora` | Training mode. |
| `--patch_size` | `14` | ViT patch size. |
| `--h5_root` | — | Root directory containing `.h5` patch files. |
| `--data_path` | — | Path to the JSON index file. |
| `--output_dir` | — | Directory for checkpoints and logs. |
| `--batch_size_per_gpu` | `48` | Per-GPU batch size. |
| `--local_crops_size` | `96` | Pixel size of local crops for multi-crop. |
| `--epochs` | `100` | Total training epochs. |
| `--lr` | `0.0005` | Peak learning rate (scaled by batch size relative to 256). |
| `--out_dim` | `65536` | DINO head output dimension. |
| `--saveckp_freq` | `20` | Save a checkpoint every N epochs. |

### Architecture & Training Mode

Only **LoRA** (Low-Rank Adaptation) is used for fine-tuning. The LoRA config:

- Rank: `r=8`, alpha: `16`
- Target modules: `attn.qkv`, `attn.proj`
- Dropout: `0.1`

This keeps the majority of backbone weights frozen and trains only the low-rank adapter parameters.

### Output

- Checkpoints are saved to `--output_dir` as `checkpoint{epoch:04}.pth`.
- Training metrics (loss, LR, weight decay) are logged to [Weights & Biases](https://wandb.ai) if `--wandb_proj_name` is set.
- Set `--wandb_mode offline` to disable online sync.

---

## 2. Downstream - Classification

BRAVE includes an Attention-Based Multiple Instance Learning (ABMIL) classifier for slide-level prediction tasks using pre-extracted patch features.

### Environment Setup

```bash
conda env create -f classification/environment.yml
conda activate brave-cls
```

### Data Preparation

#### Feature files

Pre-extract patch features from your WSIs using a foundation model (e.g., Virchow2 from the pretrained model above). Features for each slide should be saved as a `.pt` file (a 2D PyTorch tensor of shape `[N_patches, D]`):

```
/path/to/features/DATASET_NAME/
  pt_files/
    virchow2/
      slide_001.pt
      slide_002.pt
      ...
```

#### datasets.xlsx

Edit `classification/splits/datasets.xlsx` (sheet: **"feature status"**, header on row 1). Each row maps a dataset name to its feature root:

| Dataset | Feature Path |
|---|---|
| TCGA | /path/to/features/TCGA/ |
| ... | ... |

#### Study Excel file

Each classification task is defined by an Excel file in `classification/excels/`. See `TCGA-SurPost_Breast_PathSubtype.xlsx` as an example. Required columns:

| Column | Description |
|---|---|
| `dataset` | Dataset name (must match a row in `datasets.xlsx`) |
| `case` | Case/patient identifier |
| `slide` | Slide filename (`.svs` or `.ndpi`); multiple slides per case separated by `/` |
| `class` | Human-readable class name, such as `IDC` or `ILC` |
| `label` | Integer class index |
| `split` | `train`, `val`, or `test` |

For `classification/excels/TCGA-SurPost_Breast_PathSubtype.xlsx`, the actual schema is:

| dataset | case | slide | class | label | split |
|---|---|---|---|---|---|
| `TCGA-BRCA` | `TCGA-AO-A03T` | `TCGA-AO-A03T-01Z-00-DX1.8B75E203-6DC8-4AC9-A256-6FCB818DA0DD` | `IDC` | `0` | `test` |
| `TCGA-BRCA` | `TCGA-AC-A3W6` | `TCGA-AC-A3W6-01Z-00-DX1.88CC534C-F032-4E5D-9CC4-4BB50AA46880` | `ILC` | `1` | `test` |

The file contains 1053 rows in total, with splits `train`, `val`, and `test`, and labels `0` and `1`.

#### Example: run the TCGA subtype demo

Suppose your pre-extracted features are stored at:

```text
/data/pathology_features/TCGA-BRCA/
  pt_files/
    virchow2/
      TCGA-AO-A03T-01Z-00-DX1.8B75E203-6DC8-4AC9-A256-6FCB818DA0DD.pt
      TCGA-AC-A3W6-01Z-00-DX1.88CC534C-F032-4E5D-9CC4-4BB50AA46880.pt
      ...
```

Then make sure `classification/splits/datasets.xlsx` contains a row like:

| Dataset | Feature Path |
|---|---|
| `TCGA-BRCA` | `/data/pathology_features/TCGA-BRCA` |

You can launch the provided subtype classification example with the repository classification environment:

```bash
conda activate brave-cls
CUDA_VISIBLE_DEVICES=0 python classification/main.py \
    --study TCGA-SurPost_Breast_PathSubtype \
    --feature virchow2 \
    --seed 0 \
    --all_datasets classification/splits/datasets.xlsx \
    --excel_file classification/excels/TCGA-SurPost_Breast_PathSubtype.xlsx \
    --model ABMIL \
    --num_epoch 50 \
    --early_stop 10 \
    --wandb_mode offline
```

This command trains an ABMIL classifier on the `train` split, monitors the `val` split, and reports final metrics on the `test` split defined in the example Excel file.

### Training

```bash
bash classification/run.sh
```

Or directly:

```bash
CUDA_VISIBLE_DEVICES=0 python classification/main.py \
    --study TCGA-SurPost_Breast_PathSubtype \
    --feature virchow2 \
    --seed 0 \
    --all_datasets classification/splits/datasets.xlsx \
    --excel_file classification/excels/TCGA-SurPost_Breast_PathSubtype.xlsx \
    --model ABMIL \
    --num_epoch 50 \
    --early_stop 10 \
    --wandb_proj_name my_project
```

#### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--study` | — | Experiment name (used for output directory naming). |
| `--feature` | — | Feature extractor name (must match column in `datasets.xlsx`). |
| `--all_datasets` | — | Path to `datasets.xlsx` mapping datasets to feature roots. |
| `--excel_file` | — | Path to the task Excel file with case/slide/label/split columns. |
| `--model` | `ABMIL` | MIL aggregation model. |
| `--seed` | `1` | Random seed. |
| `--num_epoch` | `50` | Maximum training epochs. |
| `--early_stop` | `None` | Early stopping patience (epochs). |
| `--lr` | `2e-4` | Learning rate. |
| `--wandb_proj_name` | — | W&B project name. Set `--wandb_mode offline` to disable sync. |

### Output

Results are saved to `./results/{study}/{feature}/{model}-{seed}-{timestamp}/`:
- `checkpoint_best.pth` — best model by validation Macro AUC
- `result.json` — final test metrics (Macro AUC, Weighted AUC, ACC, F1)

---

## 3. Downstream - Survival

BRAVE also includes a survival prediction pipeline based on ABMIL with discrete-time survival loss and 5-fold cross-validation.

### Environment Setup

Create an environment first, then install the requirements:

```bash
conda create -n brave-survival python=3.10 -y
conda activate brave-survival
pip install -r survival/requirements.txt
```

### Data Preparation

#### Feature files

Each slide must have a `.pt` feature file containing a tensor of shape `[N_patches, D]`. Survival training supports multiple datasets at once, so features are organized by dataset name:

```text
/path/to/features/
  DATASET_A/
    pt_files/
      virchow2/
        slide_001.pt
        slide_002.pt
  DATASET_B/
    pt_files/
      virchow2/
        slide_101.pt
        ...
```

#### H5 coordinate files

For each slide there must also be a matching `.h5` file with a `coords` dataset. The loader uses these coordinates together with the patch features:

```text
/path/to/h5/
  DATASET_A/
    slide_001.h5
    slide_002.h5
  DATASET_B/
    slide_101.h5
    ...
```

#### Survival split Excel

Prepare an Excel file like `survival/splits/public_example.xlsx`. Required columns are:

| Column | Description |
|---|---|
| `dataset` | Dataset name. Must match a key in `--pt_roots` and `--h5_roots`. |
| `case` | Case or patient identifier. |
| `slide` | Slide ID. Multiple slides for one case should be separated by `/`. File extensions are optional. |
| `status` | Event indicator. `1` means event observed, `0` means censored. |
| `time (months)` | Survival time in months. |
| `Fold 0` ... `Fold 4` | Split assignment for each fold, typically `train`, `validation`, or `test`. |

The code discretizes `time (months)` into 4 bins based on uncensored cases and trains a 4-class discrete-time survival model.

### Training

The `--pt_roots` and `--h5_roots` arguments accept either an inline JSON object or a path to a JSON file. Each mapping must use dataset names from the Excel file as keys.

Run the provided script:

```bash
bash survival/run.sh
```

Or launch training directly:

```bash
CUDA_VISIBLE_DEVICES=0 python survival/main_kfold.py \
    --model ABMIL \
    --study TCGA-Survival \
    --feature virchow2 \
    --excel_file survival/splits/public_example.xlsx \
    --pt_roots '{"TCGA":"/path/to/features/TCGA/pt_files/virchow2"}' \
    --h5_roots '{"TCGA":"/path/to/h5/TCGA"}' \
    --num_epoch 30 \
    --folds 5 \
    --batch_size 1 \
    --seed 1 \
    --wandb_proj_name my_project \
    --wandb_exp_name TCGA-Survival-virchow2-s1
```

If you prefer JSON files for the root mappings:

```json
{
  "TCGA": "/path/to/features/TCGA/pt_files/virchow2",
  "CPTAC": "/path/to/features/CPTAC/pt_files/virchow2"
}
```

Then pass the file path instead:

```bash
python survival/main_kfold.py \
    --excel_file survival/splits/public_example.xlsx \
    --pt_roots /path/to/pt_roots.json \
    --h5_roots /path/to/h5_roots.json \
    --study multi_dataset_survival \
    --feature virchow2 \
    --model ABMIL
```

#### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--excel_file` | — | Excel file with survival labels and fold assignments. |
| `--pt_roots` | — | JSON object or JSON file mapping dataset names to `.pt` feature directories. |
| `--h5_roots` | — | JSON object or JSON file mapping dataset names to `.h5` coordinate directories. |
| `--study` | — | Experiment name used in the results directory. |
| `--feature` | — | Feature extractor name used only for bookkeeping in output paths. |
| `--model` | `ABMIL` | Survival model. |
| `--folds` | `5` | Number of cross-validation folds. |
| `--k_start` | `-1` | First fold to run. Useful for partial CV runs. |
| `--k_end` | `-1` | End fold index (exclusive). |
| `--num_epoch` | `30` | Maximum training epochs per fold. |
| `--lr` | `2e-4` | Learning rate. |
| `--weight_decay` | `1e-5` | Weight decay. |
| `--loss` | `nll_surv` | Discrete-time survival loss. |
| `--resume` | `None` | Checkpoint path, or `;`-separated checkpoint paths for evaluation across folds. |
| `--evaluate` | `False` | Run evaluation only. |
| `--wandb_proj_name` | `no-specific-proj` | W&B project name. |
| `--wandb_mode` | `None` | Set to `offline` to disable online sync. |

### Output

Results are written to `survival/results/seed_{seed}/{study}/[{model}]/[{feature}]-[{timestamp}]/`.

Each fold gets its own subdirectory:

- `fold_{k}/model_best_{epoch}.pth.tar` — best checkpoint selected by validation C-index
- `result.csv` — aggregated cross-validation results

During training, the code logs train/validation/test loss and C-index to Weights & Biases.

---

## System Requirements

### Pretraining Pipeline

- Recommended repository environment: `brave`
- Recommended setup file: `pretrain/environment.yml`
- Tested operating system: Ubuntu 22.04.3 LTS
- Tested Python: 3.9.23
- Tested CUDA: 12.8
- Tested hardware: 8 × NVIDIA H800
- Dependencies:
  - `torch==2.7.1`
  - `torchvision==0.22.1`
  - `timm==1.0.19`
  - `peft==0.17.0`
  - `loralib==0.1.2`
  - `h5py==3.14.0`
  - `wandb==0.21.0`
  - `numpy==1.26.4`
  - `Pillow==11.3.0`
  - `huggingface-hub==0.34.3`
  - `xformers==0.0.31.post1`
  - `submitit==1.5.3`
  - `transformers==4.54.1`
  - `torchmetrics==0.10.3`


### Classification Pipeline

- Recommended repository environment: `brave-cls`
- Recommended setup file: `classification/environment.yml`
- Tested operating system: Ubuntu 22.04.5 LTS
- Tested Python: 3.13.2
- Tested CUDA runtime: 11.8
- Tested hardware: single NVIDIA GeForce RTX 3090
- Dependencies in the repository environment:
  - `torch==2.6.0+cu118`
  - `torchvision==0.21.0+cu118`
  - `torchmetrics==1.8.0`
  - `h5py==3.14.0`
  - `numpy==2.1.2`
  - `openpyxl==3.1.5`
  - `pandas==2.3.1`
  - `pillow==11.0.0`
  - `tqdm==4.67.1`
  - `wandb==0.21.0`

### Survival Pipeline

- Recommended repository environment: `brave-survival`
- Recommended setup: create a Python 3.10 environment, then install `survival/requirements.txt`
- Tested operating system: Ubuntu 22.04.5 LTS
- Tested Python: 3.10.12
- Tested hardware: single NVIDIA GeForce RTX 3090
- Dependencies from `survival/requirements.txt`:
  - `apex==0.9.10dev`
  - `einops==0.8.1`
  - `fairscale==0.4.13`
  - `fairseq==0.12.2`
  - `flash_attn==2.8.3`
  - `h5py==3.13.0`
  - `huggingface_hub==0.30.2`
  - `matplotlib==3.10.7`
  - `monai==1.5.1`
  - `numpy==2.3.4`
  - `openslide_python==1.4.2`
  - `pandas==2.3.3`
  - `Pillow==12.0.0`
  - `psutil==7.0.0`
  - `PyYAML==6.0.2` or `PyYAML==6.0.3`
  - `scikit_learn==1.7.2`
  - `scikit_survival==0.24.1`
  - `scipy==1.16.3`
  - `skimage==0.0`
  - `timm==1.0.22`
  - `torch==2.6.0`
  - `torchmetrics==1.7.1`
  - `torchvision==0.21.0`
  - `tqdm==4.67.1`
  - `xformers==0.0.33`

### Hardware Requirements

- No proprietary or custom hardware is required.
- A CUDA-capable GPU is required for practical training and inference.
- Classification and survival support single-GPU execution.
- Multi-GPU hardware is recommended for pretraining.

### Typical Install Time

- Reference machine for classification and survival: Ubuntu 22.04.5 LTS, Intel Xeon Gold 5220R, about 1.0 TiB RAM, and a single NVIDIA GeForce RTX 3090 24 GB GPU.
- Reference machine for pretraining: Ubuntu 22.04.3 LTS with 8 × NVIDIA H800 GPUs.
- Approximate install time is typically 5-15 minutes for the classification environment, 10-20 minutes for the survival environment, and 10-30 minutes for the pretraining environment.
- Actual install time depends on network speed, package mirrors, and whether packages are already cached.

### Expected Demo Runtime

- On the classification reference machine above, the demo with pre-extracted features typically takes about 20-60 minutes.
- Survival runtime varies substantially with dataset size and fold count; a small validation-style run is typically on the order of tens of minutes, while a full 5-fold run can take several hours.
- On the pretraining reference machine above, as a reference, 10,000 iterations take about 3.1-3.6 hours.

Full dependency specifications are provided in the corresponding environment files.


---

## Environments Summary

| Component | Conda env | Environment file |
|---|---|---|
| Pretraining (BRAVE) | `brave` | `pretrain/environment.yml` |
| Classification (ABMIL) | `brave-cls` | `classification/environment.yml` |
| Survival (ABMIL) | `brave-survival` | `survival/requirements.txt` |

Install either environment with:

```bash
conda env create -f <environment_file>
```

## License and Terms of Tuse
ⓒ SmartXLab. This model and associated code are released under the [CC-BY-NC-ND 4.0]((https://creativecommons.org/licenses/by-nc-nd/4.0/deed.en)) license and may only be used for non-commercial, academic research purposes with proper attribution. Any commercial use, sale, or other monetization of the BRAVE model and its derivatives, which include models trained on outputs from the BRAVE model or datasets created from the BRAVE model, is prohibited and reguires prior approval.


If you have any question, feel free to email [Yingxue XU](yxueb@connect.ust.hk).
