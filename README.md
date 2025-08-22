# PathPT (Pathology Prompt-Tuning)
The official code for **"Boosting Pathology Foundation Models via Few-shot Prompt-tuning for Rare Cancer Subtyping"**

[Preprint]() | [Cite](#reference)

---

**Abstract:** Rare cancers comprise 20–25% of all malignancies but face major diagnostic challenges due to limited expert availability—especially in pediatric oncology, where they represent over 70% of cases. While pathology vision-language (VL) foundation models show promising zero-shot capabilities for common cancer subtyping, their clinical performance for rare cancers remains limited. Existing multi-instance learning (MIL) methods rely only on visual features, overlooking cross-modal knowledge and compromising interpretability critical for rare cancer diagnosis.

To address this limitation, we propose **PathPT**, a novel framework that aims to fully harnesses the potential of pre-trained vision-language models via spatially-aware visual aggregation and task-specific prompt tuning. Unlike conventional MIL, PathPT converts WSI-level supervision into fine-grained tile-level guidance by leveraging VL models’ zero-shot abilities, thereby preserving localization on cancerous regions and enabling cross-modal reasoning through prompts aligned with histopathological semantics. We benchmark PathPT on eight rare cancer datasets (four adult, four pediatric) spanning 56 subtypes and 2,910 WSIs, as well as three common cancer datasets, evaluating four state-of-the-art VL models and four MIL frameworks under three few-shot settings. Results show that PathPT consistently delivers superior performance, achieving substantial gains in subtyping accuracy and cancerous region grounding ability. This work advances AI-assisted diagnosis for rare cancers, offering a scalable solution for improving subtyping accuracy in settings with limited access to specialized expertise.

<img src="resources/teaser.png" alt="workflow" width="800" />


## Quick Start
1. Download base model KEEP from [KEEP](https://huggingface.co/Astaxanthin/KEEP) and place the model folder into ./base_models
2. Download feature of TCGA-UCS extracted using KEEP at [UCS-KEEP-feature](https://drive.google.com/file/d/1RNSIINkumfhiyqwL82hUXALCtdyPhbC3/view?usp=sharing) and place the unziped folder into ./features/keep/ucs/h5_files
3. Create a conda env
 ```bash
conda create -n pathpt python=3.8 -y
conda activate pathpt
pip install -r requirements.txt
```
4. Run train.py

## Customizing


## Benchmark


## Acknowledgment
The project was built on top of repositories such as [CLAM](https://github.com/mahmoodlab/CLAM), [CoOp](https://github.com/KaiyangZhou/CoOp) and [TransMIL](https://github.com/szc19990412/TransMIL). We thank the authors and developers for their contribution.

## Reference
If you find our work useful in your research, please consider citing our [paper]():





<img src=resources/logo.png> 
