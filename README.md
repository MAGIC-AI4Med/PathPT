# PathPT (Pathology Prompt-Tuning)
The official code for **"Boosting Pathology Foundation Models via Few-shot Prompt-tuning for Rare Cancer Subtyping"**

[Preprint](https://arxiv.org/abs/2508.15904) | [Cite](#reference)

---

**Abstract:** Rare cancers comprise 20–25% of all malignancies but face major diagnostic challenges due to limited expert availability—especially in pediatric oncology, where they represent over 70% of cases. While pathology vision-language (VL) foundation models show promising zero-shot capabilities for common cancer subtyping, their clinical performance for rare cancers remains limited. Existing multi-instance learning (MIL) methods rely only on visual features, overlooking cross-modal knowledge and compromising interpretability critical for rare cancer diagnosis.

To address this limitation, we propose **PathPT**, a novel framework that aims to fully harness the potential of pre-trained vision-language models via spatially-aware visual aggregation and task-specific prompt tuning. Unlike conventional MIL, PathPT converts WSI-level supervision into fine-grained tile-level guidance by leveraging VL models’ zero-shot abilities, thereby preserving localization on cancerous regions and enabling cross-modal reasoning through prompts aligned with histopathological semantics. We benchmark PathPT on eight rare cancer datasets (four adult, four pediatric) spanning 56 subtypes and 2,910 WSIs, as well as three common cancer datasets, evaluating four state-of-the-art VL models and four MIL frameworks under three few-shot settings. Results show that PathPT consistently delivers superior performance, achieving substantial gains in subtyping accuracy and cancerous region grounding ability. This work advances AI-assisted diagnosis for rare cancers, offering a scalable solution for improving subtyping accuracy in settings with limited access to specialized expertise.

---

## Key Insights
<img src="resources/teaser.png" alt="workflow" width="800" />

**PathPT** introduces a novel prompt-tuning framework that enhances pathology foundation models for rare cancer subtyping by fully leveraging pre-trained vision-language capabilities.

**🔹 Cross-modal Knowledge Integration**: Unlike conventional MIL methods, PathPT harnesses semantic knowledge embedded in text encoders through prompt learning, enabling cross-modal reasoning.

**🔹 Spatially-Aware Visual Aggregation**: We design a spatial-aware module that enhances the locality of visual patch features, preserving crucial spatial relationships and contextual information.

**🔹 Fine-grained Interpretable Grounding**: By leveraging foundation models' zero-shot capabilities, PathPT converts WSI-level supervision into fine-grained tile-level guidance, achieving superior localization on cancerous regions with enhanced interpretability compared to traditional approaches.

<img src="resources/visualization.png" alt="workflow" width="800" />


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
Want to use your custom pathology datasets or other foundation models? Coming soon.


## Benchmark
We benchmarked 4 MILs and PathPT on 11 datasets, covering 4 rare adult cancers, 4 rare pediatric cancers, and 3 common cancers, based on foundation models: PLIP, MUSK, CONCH, and KEEP.

<img src="resources/benchmark.png" alt="workflow" width="800" />

Results demonstrate PathPT achieves superior performance over traditional MIL frameworks. Detailed results and analysis can be found in our [paper](https://arxiv.org/abs/2508.15904).


## Acknowledgment
The project was built on top of repositories such as [CLAM](https://github.com/mahmoodlab/CLAM), [CoOp](https://github.com/KaiyangZhou/CoOp) and [TransMIL](https://github.com/szc19990412/TransMIL). We thank the authors and developers for their contribution.

## Reference
If you find our work useful in your research, please consider citing our [paper](https://arxiv.org/abs/2508.15904):

```
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




<img src=resources/logo.png> 
