import argparse
import os
import pandas as pd


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_exp', type=str, default=4)
    parser.add_argument('--inference_split', type=str, default='test')
    
    #parser.add_argument('--inference', type=str, default='modality')
    #parser.add_argument('--inference', type=str, default='location')
    parser.add_argument('--inference', type=str, default='joint')

    args = parser.parse_args()
    n_exp = args.n_exp
    inference_split = args.inference_split
    inference_label = args.inference

    path = "MedPix-2-0/"

    with open(f"{path}experiments/{n_exp}/results-{inference_split}-{inference_label}.txt", "r") as f:
        lines = f.readlines()

    exp_info = eval(lines[0])
    samples = [tuple(line[:-1].split('\t')) for line in lines[2:]]

    if exp_info['inference'] == 'joint':
        results = {
            'correct_samples_modality': 0,
            'correct_samples_location': 0,
            'correct_samples': 0,
            'tot_samples': 0,
        }
    else:
        results = {
            'correct_samples': 0,
            'tot_samples': 0,
        }

    for sample in samples:
        if exp_info['inference'] == 'joint':
            if eval(sample[3])[0] in sample[4]:
                results['correct_samples_modality'] += 1
            if eval(sample[3])[1] in sample[4]:
                results['correct_samples_location'] += 1
            if eval(sample[3])[1] in sample[4] and eval(sample[3])[0] in sample[4]:
                results['correct_samples'] += 1
        else:
            if sample[3] in sample[4]:
                results['correct_samples'] += 1
        results['tot_samples'] += 1
    
    if exp_info['inference'] == 'joint':
        print(f"Correct samples modality found {results['correct_samples_modality']} over {results['tot_samples']} samples")
        print(f"Accuracy: {results['correct_samples_modality']/results['tot_samples']*100:.2f}%")
        print(f"Correct samples location found {results['correct_samples_location']} over {results['tot_samples']} samples")
        print(f"Accuracy: {results['correct_samples_location']/results['tot_samples']*100:.2f}%")
    
    print(f"Correct samples found {results['correct_samples']} over {results['tot_samples']} samples")
    print(f"Accuracy: {results['correct_samples']/results['tot_samples']*100:.2f}%")

    if exp_info['inference'] == 'joint':
        if not os.path.exists(f'{path}experiments/overview-{inference_split}-{exp_info["inference"]}.txt'):
            with open(f'{path}experiments/overview-{inference_split}-{exp_info["inference"]}.txt', 'w') as f:
                f.write(f'N exp\tInference Split\tEmb model\tLLM model\tVisual model\tFlamingo version\tRAG\tInference Label\tExact matches modality\tExact matches location\tExact matches\tTot samples\tAccuracy modality\tAccuracy location\tAccuracy\n')
        
        with open(f'{path}experiments/overview-{inference_split}-{exp_info["inference"]}.txt', 'a') as f:
            f.write(f'{exp_info["n_exp"]}\t{exp_info["inference_split"]}\t{exp_info["emb_model"]}\t{exp_info["llm_model"]}\t{exp_info["visual_encoder"]}\t{exp_info["flamingo_v"]}\t{exp_info["rag"]}\t{exp_info["inference"]}\t{results["correct_samples_modality"]}\t{results["correct_samples_location"]}\t{results["correct_samples"]}\t{results["tot_samples"]}\t{results["correct_samples_modality"]/results["tot_samples"]*100:.2f}%\t{results["correct_samples_location"]/results["tot_samples"]*100:.2f}%\t{results["correct_samples"]/results["tot_samples"]*100:.2f}%\n')
    else:
        if not os.path.exists(f'{path}experiments/overview-{inference_split}.txt'):
            with open(f'{path}experiments/overview-{inference_split}.txt', 'w') as f:
                f.write(f'N exp\tInference Split\tEmb model\tLLM model\tVisual model\tFlamingo version\tRAG\tInference Label\tExact matches\tTot samples\tAccuracy\n')
        
        with open(f'{path}experiments/overview-{inference_split}.txt', 'a') as f:
            f.write(f'{exp_info["n_exp"]}\t{exp_info["inference_split"]}\t{exp_info["emb_model"]}\t{exp_info["llm_model"]}\t{exp_info["visual_encoder"]}\t{exp_info["flamingo_v"]}\t{exp_info["rag"]}\t{exp_info["inference"]}\t{results["correct_samples"]}\t{results["tot_samples"]}\t{results["correct_samples"]/results["tot_samples"]*100:.2f}%\n')
