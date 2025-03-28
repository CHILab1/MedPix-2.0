# MedPix 2.0 with DR-Minerva and Knowledge Graph
MedPix 2.0: A Comprehensive Multimodal Biomedical Data set for Advanced AI Applications with Retrieval Augmented Generation and Knowledge Graphs

Code for training and evaluation of DR-Minerva enhanced with knowledge graph

File in this folder, listed according to their execution order:

- `gen_template_kg.py` for generating documents for KG creation 
- `gen_kg.py` for KG generation
- `gen-questions.py` for generating the QA pairs and answers
- `inference_KG.py` for inference the KG
- `evaluate-csv.py` for evaluate the obtained predictions
- `utils.py` for utility functions

Please cite our work as follows:

```
@misc{siragusa2025medpix20comprehensivemultimodal,
      title={{MedPix 2.0: A Comprehensive Multimodal Biomedical Data set for Advanced AI Applications with Retrieval Augmented Generation and Knowledge Graphs}}, 
      author={Irene Siragusa and Salvatore Contino and Massimo La Ciura and Rosario Alicata and Roberto Pirrone},
      year={2025},
      eprint={2407.02994},
      archivePrefix={arXiv},
      primaryClass={cs.DB},
      url={https://arxiv.org/abs/2407.02994}, 
}
```
