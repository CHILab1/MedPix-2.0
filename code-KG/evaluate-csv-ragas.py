from ragas.metrics import AnswerCorrectness
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from datasets import Dataset
from ragas import evaluate
import nest_asyncio
import pandas as pd
import torchvision
import argparse
import os

if __name__ == '__main__':

    torchvision.disable_beta_transforms_warning()

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_exp', type=int, default=3)
    parser.add_argument('--inference_split', type=str, default='test')
    
    args = parser.parse_args()

    n_exp = args.n_exp
    inference_split = args.inference_split
    
    path = 'MedPix-2-0/KG/'
    
    with open(f"{path}experiments/{n_exp}/results-{inference_split}.txt", "r") as f:
        lines = f.readlines()
    res = [eval(x) for x in lines[1:]]

    with open(f"{path}experiments/{n_exp}/results-{inference_split}-mid.txt", "r") as f:
        lines = f.readlines()
    res_mid = [eval(x) for x in lines[1:]]

    with open(f"{path}experiments/qa-pairs.txt", "r") as f:
        lines = f.readlines()
    golden_dataset = [eval(x) for x in lines]

    results_simple = []
    questions_simple = [element['Query_Simple'] for element in golden_dataset]
    answers_simple = [element['Simple'] for element in res]
    
    results_mid_complex = []
    questions_mid_complex = [element['Query_Mid_complex'] for element in golden_dataset]
    answers_mid_complex = [element['Mid_complex'] for element in res_mid]
    
    results_complex = []
    questions_complex = [element['Query_Complex'] for element in golden_dataset]
    answers_complex = [element['Complex'] for element in res]
    
    golden = [element['Golden'] for element in golden_dataset]

    nest_asyncio.apply()
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    metrics = [AnswerCorrectness()]

    # Converti output_pairs in un formato compatibile con Hugging Face Dataset 

    for idx, res_mid_sample in enumerate(res_mid):
        if res_mid_sample['U_id'] != res[idx]['U_id']:
            print('Error: U_id does not match')
    
    for idx, golden_sample in enumerate(golden_dataset):
        if golden_sample['U_id'] != res[idx]['U_id']:
            print('Error: U_id does not match')
        

    for idx, golden_sample in enumerate(golden_dataset):

        dataset_dict_simple = {
            "question": [questions_simple[idx]],
            "ground_truth": [golden[idx]],
            "answer": [answers_simple[idx]],
        } 

        hf_dataset_simple = Dataset.from_dict(dataset_dict_simple)
        result_simple = evaluate(dataset=hf_dataset_simple, metrics=metrics, raise_exceptions=False)

        dataset_dict_mid_complex = {
            "question": [questions_mid_complex[idx]],
            "ground_truth": [golden[idx]],
            "answer": [answers_mid_complex[idx]],
        } 
        hf_dataset_mid_complex = Dataset.from_dict(dataset_dict_mid_complex)
        result_mid_complex = evaluate(dataset=hf_dataset_mid_complex, metrics=metrics, raise_exceptions=False, llm=evaluator_llm)


        dataset_dict_complex = {
            "question": [questions_complex[idx]],
            "ground_truth": [golden[idx]],
            "answer":[ answers_complex[idx]],
        } 
        hf_dataset_complex = Dataset.from_dict(dataset_dict_complex)
        result_complex = evaluate(dataset=hf_dataset_complex, metrics=metrics, raise_exceptions=False, llm=evaluator_llm)

        res_simple = (golden_sample['U_id'], golden_sample['img_name'], golden_sample['Modality'], golden_sample['Location'], golden_sample['Location-category'], result_simple['answer_correctness'][0])

        with open(f'{path}experiments/{n_exp}/RAGAS-results-{inference_split}-simple.txt', 'a') as f:
            f.write(str(res_simple))
            f.write('\n')

        results_simple.append(res_simple)

        res_mid_complex = (golden_sample['U_id'], golden_sample['img_name'], golden_sample['Modality'], golden_sample['Location'], golden_sample['Location-category'], result_mid_complex['answer_correctness'][0])

        with open(f'{path}experiments/{n_exp}/RAGAS-results-{inference_split}-mid_complex.txt', 'a') as f:
            f.write(str(res_mid_complex))
            f.write('\n')

        results_mid_complex.append(res_mid_complex)
        
        res_complex = (golden_sample['U_id'], golden_sample['img_name'], golden_sample['Modality'], golden_sample['Location'], golden_sample['Location-category'], result_complex['answer_correctness'][0])

        with open(f'{path}experiments/{n_exp}/RAGAS-results-{inference_split}-complex.txt', 'a') as f:
            f.write(str(res_complex))
            f.write('\n')

        results_complex.append(res_complex)

    df_simple = pd.DataFrame(results_simple, columns=['U_id', 'img_name', 'Modality', 'Location', 'Location-category', 'Aswer-Correctness'])
    
    df_mid_complex = pd.DataFrame(results_mid_complex, columns=['U_id', 'img_name', 'Modality', 'Location', 'Location-category', 'Aswer-Correctness'])

    df_complex = pd.DataFrame(results_complex, columns=['U_id', 'img_name', 'Modality', 'Location', 'Location-category', 'Aswer-Correctness'])

    df_simple.to_csv(f'{path}experiments/{n_exp}/RAGAS-results-{inference_split}-simple-complete.csv')
    df_mid_complex.to_csv(f'{path}experiments/{n_exp}/RAGAS-results-{inference_split}-mid_complex-complete.csv')
    df_complex.to_csv(f'{path}experiments/{n_exp}/RAGAS-results-{inference_split}-complex-complete.csv')

    print('end', n_exp)