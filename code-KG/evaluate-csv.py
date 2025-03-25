import pandas as pd
import evaluate
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--n_exp', type=int, default=3)
    parser.add_argument('--inference_split', type=str, default='test')
    
    args = parser.parse_args()

    device = args.device
    n_exp = args.n_exp
    inference_split = args.inference_split
    
    path = 'MedPIx-2-0/KG/'
    
    with open(f"{path}experiments/{n_exp}/results-{inference_split}.txt", "r") as f:
        lines = f.readlines()

    res = [eval(x) for x in lines[1:]]

    bleu = evaluate.load('bleu')
    rouge = evaluate.load('rouge')
    r_metrics = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
    bertscore = evaluate.load('bertscore')
    b_scores = ['precision', 'recall', 'f1']
    meteor = evaluate.load('meteor')
    perplexity = evaluate.load("perplexity", module_type="metric")

    results_simple = []
    results_complex = []

    for r in res:
        bleu_simple = bleu.compute(predictions=[r['Simple']], references=[r['Golden']])['bleu']
        bleu_complex = bleu.compute(predictions=[r['Complex']], references=[r['Golden']])['bleu']
        
        rouge_simple = rouge.compute(predictions=[r['Simple']], references=[r['Golden']])
        rouge_complex = rouge.compute(predictions=[r['Complex']], references=[r['Golden']])

        rouge1_simple = rouge_simple['rouge1']
        rouge2_simple = rouge_simple['rouge2']
        rougeL_simple = rouge_simple['rougeL']
        rougeLsum_simple = rouge_simple['rougeLsum']

        rouge1_complex = rouge_complex['rouge1']
        rouge2_complex = rouge_complex['rouge2']
        rougeL_complex = rouge_complex['rougeL']
        rougeLsum_complex = rouge_complex['rougeLsum']
        
        bert_simple = bertscore.compute(predictions=[r['Simple']], references=[r['Golden']],lang='en')
        bert_complex = bertscore.compute(predictions=[r['Complex']], references=[r['Golden']],lang='en')

        bert_simple_precision = bert_simple['precision'][0]
        bert_simple_recall = bert_simple['recall'][0]
        bert_simple_f1 = bert_simple['f1'][0]

        bert_complex_precision = bert_complex['precision'][0]
        bert_complex_recall = bert_complex['recall'][0]
        bert_complex_f1 = bert_complex['f1'][0]
        
        meteor_simple = meteor.compute(predictions=[r['Simple']], references=[r['Golden']])['meteor']
        meteor_complex = meteor.compute(predictions=[r['Complex']], references=[r['Golden']])['meteor']

        results_simple.append((r['U_id'], r['img_name'], r['Modality'], r['Location'], r['Location-category'], bleu_simple, rouge1_simple, rouge2_simple, rougeL_simple, rougeLsum_simple, bert_simple_precision, bert_simple_recall, bert_simple_f1, meteor_simple))

        results_complex.append((r['U_id'], r['img_name'], r['Modality'], r['Location'], r['Location-category'], bleu_complex, rouge1_complex, rouge2_complex, rougeL_complex, rougeLsum_complex, bert_complex_precision, bert_complex_recall, bert_complex_f1, meteor_complex))

    df_simple = pd.DataFrame(results_simple, columns=['U_id', 'img_name', 'Modality', 'Location', 'Location-category', 'BLEU', 'ROUGE1', 'ROUGE2', 'ROUGEL', 'ROUGELsum', 'BERT-Precision', 'BERT-Recall', 'BERT-F1', 'METEOR'])
    df_complex = pd.DataFrame(results_complex, columns=['U_id', 'img_name', 'Modality', 'Location', 'Location-category', 'BLEU', 'ROUGE1', 'ROUGE2', 'ROUGEL', 'ROUGELsum', 'BERT-Precision', 'BERT-Recall', 'BERT-F1', 'METEOR'])

    df_simple.to_csv(f'{path}experiments/{n_exp}/results-{inference_split}-simple.csv')
    df_complex.to_csv(f'{path}experiments/{n_exp}/results-{inference_split}-complex.csv')
