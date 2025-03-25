# DR-Minerva
DR-Minerva: a Multimodal Language Model based on Minerva for Diagnostic Information Retrieval

Code for training and evaluation of DR-Minerva 

File in this folder, listed according to their execution order:

- `RAG_construction.py` for building of the RAG module, as for template and vectorstore creation 
- `finetune-minerva.py` for fine-tuneìing textual encoder and generate lora adpters
- `merge_weights.py` for saving fine-tuned model and merges adapters
- `inference_rag_gen.py` for retrieving from vectorstore for subsequent inference
- `gen_dataset_ft_flamingo` for arranging dataset for Flamingo fine-tuning
- `train.py` for training for flamingo architecture
- `eval.py` for inference with flamingo + fine-tuned textual encoder
- `evaluate-code.py` for evaluate the obtained output in termes of accuracy
- `utils.py` for utility functions

Please cite our work as follows:

```
@inproceedings{drMinerva,
    author = {Siragusa, Irene and Contino, Salvatore and Pirrone, Roberto},
    title = {{DR-Minerva: a Multimodal Language Model based on Minerva for Diagnostic Information Retrieval}},
    booktitle="AIxIA 2024 -- Advances in Artificial Intelligence",
    year="2025",
    publisher="Springer Nature Switzerland",
    address="Cham",
    pages="288--300",
    isbn="978-3-031-80607-0"
}
```
