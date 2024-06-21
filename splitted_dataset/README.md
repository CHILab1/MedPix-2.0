### Splitted dataset

The dataset is splitted into `train`, `dev` and `test`.

Here we report the splits as jsonl file, namely `data_\*.jsonl` and `descriptions_\*.jsonl`, the splitted version of `Case_topic.json` and `Descriptions.json.`

A clinical report is considered as indivisible, i.e. images belonging to the same clinical case are in the same split.
Splits are balanced assuring a equal division as regards scanning modality and location category field the the images.
We further divided the training split in `train-1` and `train-2`, both splits are equally balanced as reported before.
`uid_list.json` provides a list version of the clinical cases per each split. 

Splits can be loaded as HuggingFace datasets:

~~~
from datasets import load_dataset, DatasetDict

path = "MedPix-2.0/splitted_dataset/"
dataset_train_1 = load_dataset("json", data_files=path+"descriptions_train_1.jsonl")
dataset_train_2 = load_dataset("json", data_files=path+"descriptions_train_2.jsonl")
dataset_train = load_dataset("json", data_files=path+"descriptions_train.jsonl")
dataset_dev = load_dataset("json", data_files=path+"descriptions_dev.jsonl")
dataset_test = load_dataset("json", data_files=path+"descriptions_test.jsonl")

dataset = DatasetDict({
    'train' : dataset_train['train'],
    'train_1' : dataset_train_1['train'],
    'train_2' : dataset_train_2['train'],
    'dev' : dataset_dev['train'],
    'test' : dataset_test['train'],
    })
~~~
