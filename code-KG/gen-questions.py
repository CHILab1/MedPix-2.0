from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import StorageContext
from llama_index.core import Settings
from tqdm import tqdm
import argparse
import pickle
import copy
import os

from utils import load_complete_dataset, load_complete_dataset_description, get_description_per_uid, get_block

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference_split', type=str, default='test')

    args = parser.parse_args()
    inference_split = args.inference_split

    path = 'MedPix-2-0/KG/'
    path_results = 'MedPix-2-0/'
    inference_label = 'joint'
    n_exp_minerva = 4

    dataset = load_complete_dataset()
    dataset_description = load_complete_dataset_description()

    # clustering descriptions per uid
    description_per_uid = get_description_per_uid(dataset_description)

    os.makedirs(f"{path}experiments/", exist_ok=True)
    with open(f"{path}experiments/qa_pairs.txt", "w") as f:
        f.write('')

    query_simple_raw = "Can you tell me which disease is most probable to be found in a patient having a {DR_Minerva_out}?"

    query_mid_complex_raw  = "Can you tell me which disease is most probable to be found in a {age}{sex} patient according to a {DR_Minerva_out}? "

    query_complex_raw = "Can you tell me which disease is most probable to be found in a {age}{sex} patient according to a {DR_Minerva_out}? Consider also the following additional information about the patient.\n{history}.?"

    results = []

    with open(f"{path_results}experiments/{n_exp_minerva}/results-{inference_split}-{inference_label}.txt", "r") as f:
        lines = f.readlines()

    exp_info = eval(lines[0])
    samples_dr_minerva = [tuple(line[:-1].split('\t')) for line in lines[2:]]

    for idx, sample in enumerate(tqdm(dataset[inference_split])):
        descriptions = description_per_uid[sample['U_id']]
        history = get_block(sample, 'History', 'Case')
        #print(history)
        disease_discussion = get_block(sample, 'Disease Discussion', 'Topic')
        title_specific = get_block(sample, 'Title', 'Case')
        title_generic = get_block(sample, 'Title', 'Topic')
        case_name = get_block(sample, 'Case Diagnosis', 'Case')
        case_names = list(set([title_generic, title_specific, case_name]))
        case_names_str = ' or '.join(x for x in case_names)

        for description in descriptions:
            age = get_block(description, 'Age', 'Description')
            sex = get_block(description, 'Sex', 'Description')
            modality = get_block(description, 'Modality', 'Description')
            location = description['Location'] 
            location_category = description['Location Category'] 

            if sex != '':
                sex = ' '+sex

            minerva_out = samples_dr_minerva[idx][4]

            query_simple = copy.deepcopy(query_simple_raw)
            query_mid_complex = copy.deepcopy(query_mid_complex_raw)
            query_complex = copy.deepcopy(query_complex_raw)

            golden = f'{age}{sex} patient suffering from {case_names_str}. {disease_discussion}'
            query_simple = query_simple.format(DR_Minerva_out=minerva_out)
            query_mid_complex = query_mid_complex.format(DR_Minerva_out=minerva_out, age=age, sex=sex)
            query_complex = query_complex.format(DR_Minerva_out=minerva_out, age=age, sex=sex, history=get_block(sample, 'History', 'Case'))

            result = {
                'idx' : idx,
                'U_id' : sample['U_id'],
                'img_name' : description['image'],
                'Golden' : golden,
                'Query_Simple' : query_simple,
                'Query_Mid_complex' : query_mid_complex,
                'Query_Complex' : query_complex,
                'Modality' : modality,
                'Location' : location,
                'Location-category' : location_category,
            }

            with open(f"{path}experiments/qa_pairs.txt", "a") as f:
                f.write(f"{result}\n")

            results.append(result)
