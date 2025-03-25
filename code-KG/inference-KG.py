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
    parser.add_argument('--llm_model', type=str, default='llama31inst')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--inference_split', type=str, default='test')
    parser.add_argument('--split_kg', type=str, default='train')
    parser.add_argument('--typology', type=str, default='hybrid')
    parser.add_argument('--relations', type=int, default=10)
    parser.add_argument('--n_exp', type=int, default=0)
    parser.add_argument('--lower_bool', action='store_true')
    parser.add_argument('--no-lower_bool', dest='lower_bool', action='store_false')
    parser.set_defaults(lower_bool=False)

    args = parser.parse_args()

    llm_model = args.llm_model
    device = args.device
    inference_split = args.inference_split
    split_kg = args.split_kg
    typology = args.typology
    relations = args.relations
    n_exp = args.n_exp
    lower_bool = args.lower_bool

    if lower_bool:
        lower = '-lower'
    else:
        lower = ''

    if llm_model == 'llama31inst':
        # path to local llama 3.1 8B inst
        llm_model_path = 'LLM/llama31inst/'

    path = 'MedPIx-2-0/KG/'
    path_results = 'MedPIx-2-0/'
    inference_label = 'joint'
    n_exp_minerva = 4

    Settings.llm = HuggingFaceLLM(model_name=llm_model_path,
                                    tokenizer_name=llm_model_path,
                                    device_map=device,
                                    generate_kwargs={
                                        "temperature": 0.00001,
                                        "no_repeat_ngram_size":2,
                                        })

    Settings.embed_model = None
    Settings.chunk_size = 4096

    #setup the storage context
    graph_store = SimpleGraphStore()
    storage_context = StorageContext.from_defaults(persist_dir=f"{path}graphs/{typology}-{split_kg}-{relations}trips{llm_model}{lower}")
    index = load_index_from_storage(storage_context)

    query_engine = index.as_query_engine(include_text=True,
                                        response_mode ="tree_summarize",
                                        embedding_mode="hybrid",
                                        similarity_top_k=5
                                        )

    dataset = load_complete_dataset()
    dataset_description = load_complete_dataset_description()

    # clustering descriptions per uid
    description_per_uid = get_description_per_uid(dataset_description)


    os.makedirs(f"{path}experiments/{n_exp}-no-inst", exist_ok=True)

    with open(f"{path}experiments/{n_exp}-no-inst/results-{inference_split}.txt", "w") as f:
        exp_info = {
            'n_exp': n_exp,
            'inference_split' : inference_split,
            'llm_model' : llm_model,
            'split' : split_kg,
            'typology' : typology,
            'relations' : relations,
            'lower_bool' : lower_bool,
        }

        f.write(f"{exp_info}\n")

    query_simple_raw = "Can you tell me which disease is most probable to be found in a patient having a {DR_Minerva_out}?"

    query_mid_complex_raw = "Can you tell me which disease is most probable to be found in a {age}{sex} patient according to a {DR_Minerva_out}?"

    query_complex_raw = "Can you tell me which disease is most probable to be found in a {age}{sex} patient according to a {DR_Minerva_out}? Consider also the following additional information about the patient.\n{history}.?"

    results = []

    with open(f"{path_results}experiments/{n_exp_minerva}/results-{inference_split}-{inference_label}.txt", "r") as f:
        lines = f.readlines()

    exp_info = eval(lines[0])
    samples_dr_minerva = [tuple(line[:-1].split('\t')) for line in lines[2:]]

    for idx, sample in enumerate(tqdm(dataset[inference_split])):
        descriptions = description_per_uid[sample['U_id']]
        history = get_block(sample, 'History', 'Case')
        disease_discussion = get_block(sample, 'Disease Discussion', 'Topic')
        title_specific = get_block(sample, 'Title', 'Case')
        title_generic = get_block(sample, 'Title', 'Topic')
        case_name = get_block(sample, 6)
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

            golden = f'{age}{sex} patient suffering from {case_names_str}. {disease_discussion}'

            query_simple = copy.deepcopy(query_simple_raw)
            query_mid_complex = copy.deepcopy(query_mid_complex_raw)
            query_complex = copy.deepcopy(query_complex_raw)

            response_simple = query_engine.query(query_simple.format(DR_Minerva_out=minerva_out))
            response_mid_complex = query_engine.query(query_mid_complex.format(DR_Minerva_out=minerva_out, age=age, sex=sex))
            response_complex = query_engine.query(query_complex.format(DR_Minerva_out=minerva_out, age=age, sex=sex, history=history))

            result = {
                'idx' : idx,
                'U_id' : sample['U_id'],
                'img_name' : description['image'],
                'Golden' : golden,
                'Simple' : response_simple.response.strip(),
                'Mid_complex' : response_mid_complex.response.strip(),
                'Complex' : response_complex.response.strip(),
                'Query_Simple' : query_simple.format(DR_Minerva_out=minerva_out),
                'Query_Mid_complex' : query_mid_complex.format(DR_Minerva_out=minerva_out, age=age, sex=sex),
                'Query_Complex' : query_complex.format(DR_Minerva_out=minerva_out, age=age, sex=sex, history=history),
                'Modality' : modality,
                'Location' : location,
                'Location-category' : location_category,
            }

            with open(f"{path}experiments/{n_exp}-no-inst/results-{inference_split}.txt", "a") as f:
                f.write(f"{result}\n")

            results.append(result)
    
    with open(f"{path}experiments/{n_exp}-no-inst/results-{inference_split}.pkl", "wb") as f:
        pickle.dump(results, f)