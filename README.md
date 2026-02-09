# 🩺 PathPT (Pathology Prompt-Tuning)

<div align="center">

**🚀 Boosting Pathology Foundation Models via Few-shot Prompt-tuning for Rare Cancer Subtyping**

[![arXiv](https://img.shields.io/badge/arXiv-2508.15904-b31b1b.svg)](https://arxiv.org/abs/2508.15904)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-3776ab.svg)](https://www.python.org/)

[📄 Paper](https://arxiv.org/abs/2508.15904) • [🔧 Quick Start](#-quick-start) • [📊 Benchmark](#-benchmark-results) • [💡 Citation](#-citation)

</div>

---

## 📋 Abstract

**🔬 The Challenge**: Rare cancers comprise 20–25% of all malignancies but face major diagnostic challenges due to limited expert availability—especially in pediatric oncology, where they represent over 70% of cases. While pathology vision-language (VL) foundation models show promising zero-shot capabilities for common cancer subtyping, their clinical performance for rare cancers remains limited.

**💡 Our Solution**: We propose **PathPT**, a novel framework that fully harnesses the potential of pre-trained vision-language models via spatially-aware visual aggregation and task-specific prompt tuning. Unlike conventional MIL methods that rely only on visual features, PathPT enables cross-modal reasoning through prompts aligned with histopathological semantics.

**📈 Impact**: Benchmarked on 8 rare cancer datasets (4 adult, 4 pediatric) spanning 56 subtypes and 2,910 WSIs, plus 3 common cancer datasets, PathPT consistently delivers superior performance with substantial gains in subtyping accuracy and cancerous region grounding ability.

---

## ✨ Key Insights

<div align="center">
<img src="resources/teaser.png" alt="PathPT Workflow" width="800" />
</div>

**PathPT** introduces a novel prompt-tuning framework that enhances pathology foundation models for rare cancer subtyping by fully leveraging pre-trained vision-language capabilities.

### 🎯 Core Innovations

**🔄 Cross-modal Knowledge Integration**: Unlike conventional MIL methods, PathPT harnesses semantic knowledge embedded in text encoders through prompt learning, enabling sophisticated cross-modal reasoning.

**🗺️ Spatially-Aware Visual Aggregation**: Our carefully designed spatial-aware module enhances the locality of visual patch features, preserving crucial spatial relationships and contextual information.

**🎯 Fine-grained Interpretable Grounding**: By leveraging foundation models' zero-shot capabilities, PathPT converts WSI-level supervision into fine-grained tile-level guidance, achieving superior localization on cancerous regions with enhanced interpretability.

<div align="center">
<img src="resources/visualization.png" alt="Visualization Results" width="800" />
</div>

---

## 🚀 Quick Start

### 📦 Installation

```bash
# Create and activate conda environment
conda create -n pathpt python=3.8 -y
conda activate pathpt
conda install openslide=3.4.1

# Install dependencies
pip install -r requirements.txt
```

### 🔧 Setup

1. **📂 Download Base Model**: Get a foundation model like [KEEP](https://huggingface.co/Astaxanthin/KEEP) and place it in `./base_models/`

2. **💾 Download Features**: Get pre-extracted features like [UCS-KEEP-features](https://drive.google.com/file/d/1RNSIINkumfhiyqwL82hUXALCtdyPhbC3/view?usp=sharing) and place in `./features/keep/ucs/h5_files/`

3. **🏃‍♂️ Run PathPT**:
   For quick start:
   ```bash
   quick_start_inference.ipynb
   ```
   For training:
   ```bash
   python train.py
   ```

> **⚠️ Note**: If you encounter issues with `Nystrom-Attention`, check out the [solution here](https://github.com/szc19990412/TransMIL/issues/33).

---

## 🧸 KidRare Dataset

We have released **KidRare**, a specialized WSI dataset focused on rare pediatric tumors, to facilitate research in computational pathology.

### 📊 Dataset Overview
- **Content**: 1,232 WSIs
- **Cancer Types**: Neuroblastoma, Nephroblastoma, Medulloblastoma, Hepatoblastoma
- **License**: CC-BY-NC-ND-3.0 (Non-commercial, Academic Research Only)

### 📥 Access & Labels

**1. Download Images**:
The dataset is hosted on Hugging Face. You need to request access and agree to the terms of use.
> 🔗 **Hugging Face Link**: [https://huggingface.co/datasets/Firehdx233/KidRare](https://huggingface.co/datasets/Firehdx233/KidRare)

**2. Get Labels & Splits**:
The label files and dataset division used in our paper can be found directly in this repository:
- 📂 **File Path**: [`multifold/dataset_division.json`](https://github.com/MAGIC-AI4Med/PathPT/blob/main/multifold/dataset_division.json)

---

## 🛠️ Customization Guide

Want to use your own datasets and foundation models? We've got you covered! 🎉

### 🤖 Base Model Setup

Download your foundation model into `./base_models/`, e.g.: [KEEP](https://huggingface.co/Astaxanthin/KEEP) [[1]](https://arxiv.org/abs/2412.13126), [CONCH](https://huggingface.co/MahmoodLab/conch) [[2]](https://www.nature.com/articles/s41591-024-02856-4), [MUSK](https://huggingface.co/xiangjx/musk) [[3]](https://www.nature.com/articles/s41586-024-08378-w), [PLIP](https://huggingface.co/vinid/plip) [[4]](https://www.nature.com/articles/s41591-023-02504-3).

> **💡 Important**: Only vision-language models with patch encoders are supported!

### 📊 Dataset Division

#### 1️⃣ Structure your data in `./dataset_division.json`:

<details>
<summary>📋 Click to see example format</summary>

```json
{
  "YOUR_DATASET": {
    "train_IDs": {
      "1": ["sample1_class1", "sample2_class1"],
      "2": ["sample1_class2", "sample2_class2"]
    },
    "test_IDs": {
      "1": ["test_sample1_class1"],
      "2": ["test_sample1_class2"]
    },
    "name2label": {
      "Class Name 1": 1,
      "Class Name 2": 2
    }
  }
}
```

</details>

#### 2️⃣ Create CSV files in `./multifold/` with columns:
`train`, `train_label`, `val`, `val_label`, `test`, `test_label`

📝 See `./multifold/dataset_csv_10shot/TCGA/UCS/fold5.csv` for reference.

### 🔍 Feature Extraction

Extract visual features from WSI patches using your foundation model:

<details>
<summary>🔧 Example with KEEP</summary>

```python
# Load your base model
model = AutoModel.from_pretrained("Astaxanthin/KEEP", trust_remote_code=True)
model.eval()

# Setup transforms
transform = transforms.Compose([
    transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# Process patch
example_patch_path = 'YOUR_PATCH_IMG_FILE'
example_patch = Image.open(example_patch_path).convert('RGB')
patch_feature = model.encode_image(transform(example_patch).unsqueeze(0))
```

</details>

**🛠️ WSI Preprocessing**: We recommend [TRIDENT](https://github.com/mahmoodlab/TRIDENT) for WSI preprocessing and patching.

**📁 File Format**: Save patch features as `.h5` files with:
- **Dataset 1**: Patch features `[N, feature_dim]`
- **Dataset 2**: Coordinates `[N, 2]`

Place files in `./features/YOUR_DATASET/YOUR_MODEL/h5_files/`

### 🔌 Model Integration

For **new foundation models**, create these template files:

| File | Purpose | 📝 Template |
|------|---------|-------------|
| `./models/PathPT_model_YOUR_MODEL.py` | Model architecture | [Template](./models/PathPT_model_YOUR_MODEL.py) |
| `./wsi_selecters/wsi_selecter_YOUR_MODEL.py` | Patch selection | [Template](./wsi_selecters/wsi_selecter_YOUR_MODEL.py) |
| `./subtyping/main_wsi_subtyping_YOUR_MODEL.py` | Training pipeline | [Template](./subtyping/main_wsi_subtyping_YOUR_MODEL.py) |

### ⚙️ Configuration

#### 1️⃣ Update `./params.py`:
```python
subtype_params['your_dataset'] = {
    'dataset_name': 'your_dataset',
    'your_model_feature_root': './features/your_dataset/your_model/h5_files',
    'patch_num': None,
    'repeats':10
    # ... other parameters
}
```
> **💡 Note**: Init pattern of learning prompts can be customized here.


#### 2️⃣ Update `./utils.py`:
```python
your_dataset_names = {
  'subtype1': ['name1', 'name2'],
  'subtype2': ['name1', 'name2'],
  'Normal': ['name1', 'name2', 'name3']
  }
```
> **💡 Note**: Subtype names used in prompts can be customized here.

#### 3️⃣ Modify `./train.py`:
```python
# Change import to your model
from subtyping.main_wsi_subtyping_YOUR_MODEL import main_subtyping as main_YOUR_MODEL
```

### 🏃‍♂️ Run Training

```bash
python train.py --model YOUR_MODEL --dataset YOUR_DATASET --shot 10
```

**📊 Monitor Progress**: Check training logs in `./logs` and `./fewshot_results`for progress and metrics!

---

## 📊 Benchmark Results

We evaluated **4 MIL methods** and **PathPT** across **11 datasets** covering:
- 🩺 4 rare adult cancers
- 👶 4 rare pediatric cancers  
- 🔬 3 common cancers

Using foundation models: **PLIP**, **MUSK**, **CONCH**, and **KEEP**.

<div align="center">
<img src="resources/benchmark.png" alt="Benchmark Overview" width="800" />
</div>

### 🏆 ERBAINS Results (Balanced Accuracy)

| **Model** | **Zero-shot** | **ABMIL** | **CLAM** | **TransMIL** | **DGRMIL** | **PathPT (Ours)** |
|:---------:|:-------------:|:---------:|:--------:|:------------:|:----------:|:-----------------:|
| **PLIP** | 0.111 | 0.419 | 0.410 | 0.488 | 0.491 | 0.251 |
| **MUSK** | 0.253 | 0.403 | 0.442 | 0.582 | 0.569 | 0.519 |
| **CONCH** | 0.204 | 0.542 | 0.549 | 0.621 | 0.621 | 0.491 |
| **KEEP** | 0.408 | 0.631 | 0.629 | 0.648 | 0.650 | **🏆 0.679** |

### 🧬 Hepatoblastoma Results (Balanced Accuracy)

| **Model** | **Zero-shot** | **ABMIL** | **CLAM** | **TransMIL** | **DGRMIL** | **PathPT (Ours)** |
|:---------:|:-------------:|:---------:|:--------:|:------------:|:----------:|:-----------------:|
| **PLIP** | 0.278 | 0.370 | 0.363 | 0.384 | 0.435 | 0.349 |
| **MUSK** | 0.395 | 0.262 | 0.378 | 0.447 | 0.436 | 0.438 |
| **CONCH** | 0.276 | 0.376 | 0.374 | 0.518 | 0.491 | **🏆 0.534** |
| **KEEP** | 0.313 | 0.383 | 0.378 | 0.492 | 0.444 | 0.524 |

### 🌊 UBC-OCEAN Results (Balanced Accuracy)

| **Model** | **Zero-shot** | **ABMIL** | **CLAM** | **TransMIL** | **DGRMIL** | **PathPT (Ours)** |
|:---------:|:-------------:|:---------:|:--------:|:------------:|:----------:|:-----------------:|
| **PLIP** | 0.320 | 0.565 | 0.570 | 0.645 | 0.630 | 0.510 |
| **MUSK** | 0.520 | 0.570 | 0.610 | 0.720 | 0.700 | 0.730 |
| **CONCH** | 0.375 | 0.590 | 0.605 | 0.710 | 0.715 | 0.790 |
| **KEEP** | 0.660 | 0.755 | 0.730 | 0.795 | 0.795 | **🏆 0.820** |

> **📈 Key Takeaway**: PathPT achieves superior performance over traditional MIL methods! For detailed analysis, check our [paper](https://arxiv.org/abs/2508.15904).

---

## 🙏 Acknowledgments

This project builds upon amazing work from the community such as [CLAM](https://github.com/mahmoodlab/CLAM), [CoOp](https://github.com/KaiyangZhou/CoOp), [TransMIL](https://github.com/szc19990412/TransMIL). Big thanks to all the authors and developers! 👏

---

## 💡 Citation

If you find our work useful, please consider citing our paper:

```bibtex
@misc{he2025boostingpathologyfoundationmodels,
      title={Boosting Pathology Foundation Models via Few-shot Prompt-tuning for Rare Cancer Subtyping}, 
      author={Dexuan He and Xiao Zhou and Wenbin Guan and Liyuan Zhang and Xiaoman Zhang and Sinuo Xu and Ge Wang and Lifeng Wang and Xiaojun Yuan and Xin Sun and Yanfeng Wang and Kun Sun and Ya Zhang and Weidi Xie},
      year={2025},
      eprint={2508.15904},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.15904}, 
}
```
<!-- 
---

<div align="center">
<img src="resources/logo.png" width="200"/>

**🌟 Star us on GitHub if this helps your research! 🌟**

[![Star History Chart](https://api.star-history.com/svg?repos=your-username/PathPT&type=Date)](https://star-history.com/#your-username/PathPT&Date)


</div> -->




