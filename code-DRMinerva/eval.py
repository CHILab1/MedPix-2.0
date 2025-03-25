from PIL import Image
from torch import cuda
import torch
from tqdm import tqdm
import argparse
import json
import os

from utils import get_block, get_description_per_uid, get_rag_context, get_static_context, inference, load_complete_dataset, load_complete_dataset_description, load_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--starting_index', type=int, default=0)
    parser.add_argument('--n_exp', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--inference_split', type=str, default='test')
    parser.add_argument('--emb_model', type=str, default='mistral')
    parser.add_argument('--llm_model', type=str, default='dr_minerva')
    parser.add_argument('--zero-shot', action='store_true')
    parser.add_argument('--no-zero-shot', dest='zero_shot', action='store_false')
    parser.set_defaults(zero_shot=False)
    parser.add_argument('--rag', action='store_true')
    parser.add_argument('--no-rag', dest='rag', action='store_false')
    parser.set_defaults(rag=False)
    
    #parser.add_argument('--inference', type=str, default='modality')
    #parser.add_argument('--inference', type=str, default='location')
    parser.add_argument('--inference', type=str, default='joint')
    
    parser.add_argument('--checkpoint_load', action='store_true')
    parser.add_argument('--no-checkpoint_load', dest='checkpoint_load', action='store_false')
    parser.set_defaults(checkpoint_load=False)
    parser.add_argument('--checkpoint_n', type=int, default=1)
    args = parser.parse_args()
    
    starting_index = args.starting_index
    n_exp = args.n_exp
    device_name = args.device
    inference_split = args.inference_split
    emb_model = args.emb_model
    llm_model = args.llm_model
    zero_shot = args.zero_shot
    rag = args.rag
    inference_label = args.inference

    if emb_model == 'mistral':
        emb_model_name = "Linq-AI-Research/Linq-Embed-Mistral"
        emb_model_str = 'linq-embed-mistral'
    else:
        emb_model == 'mistral'
        emb_model_name = "Linq-AI-Research/Linq-Embed-Mistral"
        emb_model_str = 'linq-embed-mistral'

    if llm_model == 'minerva':
        llm_model_name = "sapienzanlp/Minerva-3B-base-v1.0"
        llm_model_str = "Minerva-3B-base-v1.0"
    elif llm_model == 'dr_minerva':
        llm_model_name = "PATH/TO/dr_minerva"
        llm_model_str = "dr_minerva"
    else:
        llm_model == 'minerva'
        llm_model_name = "sapienzanlp/Minerva-3B-base-v1.0"
        llm_model_str = "Minerva-3B-base-v1.0"

    if inference_label not in ['modality', 'location', 'joint']:
        inference_label='modality'
    
    dataset = load_complete_dataset()
    dataset_description = load_complete_dataset_description()

    # clustering descriptions per uid
    description_per_uid = get_description_per_uid(dataset_description)
    path = "MedPix-2-0/"

    clip_vision_encoder_path = "ViT-L-14"
    clip_vision_encoder_pretrained = "openai"
    lang_encoder_path = llm_model_name
    tokenizer_path = llm_model_name

    device = device_name if cuda.is_available() else 'cpu'
    
    if args.checkpoint_load:
        checkpoint_path = f'MedPix-2-0/DR_minerva_flamingo/checkpoint_{args.checkpoint_n}.pt'
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        flamingo_v = f'flamingo-ft-{args.checkpoint_n}'
    else:
        checkpoint = None
        flamingo_v = 'flamingo-base'


    model, image_processor, tokenizer = load_model(clip_vision_encoder_path, clip_vision_encoder_pretrained, lang_encoder_path, tokenizer_path, device, checkpoint)
    tokenizer.padding_side = "left"
    model.eval()
    
    # inference
    os.makedirs(f"{path}experiments/{n_exp}", exist_ok=True)

    with open(f"{path}/retrieved-over-{inference_split}-history-{emb_model_str}.json", "r") as f:
        retrieved_reports = json.load(f)

    if starting_index == 0:
        with open(f"{path}experiments/{n_exp}/results-{inference_split}-{inference_label}.txt", "w") as f:

            exp_info = {
                'n_exp': n_exp,
                'inference_split' : inference_split,
                'emb_model' : emb_model_name,
                'llm_model' : llm_model_name,
                'visual_encoder' : clip_vision_encoder_path,
                'flamingo_v' : flamingo_v,
                'rag' : rag,
                'inference' : inference_label,
            }

            f.write(f"{exp_info}\n")
            f.write(f"Sample_idx\tSample_id\tImage_id\tGolden label\tPrediction\n")

    if zero_shot:
        print(f"Experiment {n_exp}: starting inference on {inference_split} split about {inference_label} w/ {llm_model_name} and {clip_vision_encoder_path} encoder w/ {flamingo_v} w/ zero-shot")
    elif rag:
        print(f"Experiment {n_exp}: starting inference on {inference_split} split about {inference_label} w/ {llm_model_name} and {clip_vision_encoder_path} encoder w/ {flamingo_v} w/ RAG over history {emb_model_str} corpus")
    else:
        print(f"Experiment {n_exp}: starting inference on {inference_split} split about {inference_label} w/ {llm_model_name} and {clip_vision_encoder_path} encoder w/ {flamingo_v} w/o RAG")
    
    with open(f"{path}experiments/{n_exp}/raw_output-{inference_split}-{inference_label}.txt", "w") as f:
        f.write(f"{'-'*50}\n")

    for idx, sample in enumerate(tqdm(dataset[inference_split])):
        if idx >= starting_index:
            
            # preparare context
            if zero_shot:
                context_str, vision_context, context_counter = '', [], 0
            elif not rag:
                context_str, vision_context, context_counter = get_static_context(dataset, path, image_processor, description_per_uid, inference_label)
            else:
                context_str, vision_context, context_counter = get_rag_context(sample, retrieved_reports, dataset, path, image_processor, description_per_uid, inference_label)

            with open(f"{path}code/prompt.json", 'r') as f:
                raw_prompt = json.load(f)

            if inference_label == 'modality':
                prompt = raw_prompt['instruction_modality']+context_str
            elif inference_label == 'location':
                prompt = raw_prompt['instruction_location']+context_str
            elif inference_label == 'joint':
                prompt = raw_prompt['instruction_joint']+context_str

            descriptions = description_per_uid[sample['U_id']]
            history = get_block(sample, 'History', 'Case')

            for description in descriptions:
                age = get_block(description, 'Age', 'Description')
                sex = get_block(description, 'Sex', 'Description')
                image = Image.open(path+'MedPix-2-0/images/'+description['image']+'.png')
                vision_query = image_processor(image).unsqueeze(0)
                if inference_label == 'modality' or inference_label == 'joint':
                    split_text = "The image is a "
                elif inference_label == 'location':
                    split_text = "The image shows a "
                text = f"<image> {age}{sex}patient. {history}\n{split_text}"
                
                raw_output = inference(vision_context + [vision_query], prompt+text, model, tokenizer, device).split(split_text)[context_counter]
                output = raw_output.replace('\n', ' <nl>')

                with open(f"{path}experiments/{n_exp}/results-{inference_split}-{inference_label}.txt", "a") as f:
                    if inference_label == 'modality':
                        f.write(f"{idx}\t{sample['U_id']}\t{description['image']}\t{description['Type']}\t{output}\n")
                    elif inference_label == 'location':
                        f.write(f"{idx}\t{sample['U_id']}\t{description['image']}\t{description['Location Category']}\t{output}\n")
                    elif inference_label == 'joint':
                        f.write(f"{idx}\t{sample['U_id']}\t{description['image']}\t['{description['Type']}', '{description['Location Category']}']\t{output}\n")
                    
                with open(f"{path}experiments/{n_exp}/raw_output-{inference_split}-{inference_label}.txt", "a") as f:
                    f.write(f"{raw_output}\n")
                    f.write(f"{'-'*50}\n")
        
    print(f"Finished inference")