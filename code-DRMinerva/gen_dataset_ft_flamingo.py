from PIL import Image
import pandas as pd
from torch import cuda
from tqdm import tqdm
import argparse
import torch
import json
import os

from utils import get_block, get_description_per_uid, get_rag_context, get_static_context, load_complete_dataset, load_complete_dataset_description, load_model

from MyDataset import MyDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_model', type=str, default='dr_minerva')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    llm_model = args.llm_model
    device_name = args.device

    emb_model = 'mistral'
    emb_model_name = "Linq-AI-Research/Linq-Embed-Mistral"
    emb_model_str = 'linq-embed-mistral'

    if llm_model == 'minerva':
        llm_model_name = "sapienzanlp/Minerva-3B-base-v1.0"
        llm_model_str = "Minerva-3B-base-v1.0"
    elif llm_model == 'dr_minerva':
        llm_model_name = "MedPix-2-0/dr_minerva"
        llm_model_str = "dr_minerva"
    else:
        llm_model == 'minerva'
        llm_model_name = "sapienzanlp/Minerva-3B-base-v1.0"
        llm_model_str = "Minerva-3B-base-v1.0"
    
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
    #device = 'cpu'

    model, image_processor, tokenizer = load_model(clip_vision_encoder_path, clip_vision_encoder_pretrained, lang_encoder_path, tokenizer_path, device)
    tokenizer.padding_side = "right"

    os.makedirs(f"{path}data_ft_flamingo/", exist_ok=True)

    # create dataloader
    # put loss and optimizer following the train.py

    inference_splits = ['train_2', 'dev', 'test']
    inference_labels = ['joint', 'modality', 'location']

    for inference_split in inference_splits:
        samples = []

        for inference_label in inference_labels:

            with open(f"{path}retrieved-over-{inference_split}-history-{emb_model_str}.json", "r") as f:
                retrieved_reports = json.load(f)

            for idx, sample in enumerate(tqdm(dataset[inference_split])):

                # no-rag w/o in context learning 
                static_context_str, static_context_vision, static_context_counter = get_static_context(dataset, path, image_processor, description_per_uid, inference_label)
                # preparare context
                context_str, context_vision, context_counter = get_rag_context(sample, retrieved_reports, dataset, path, image_processor, description_per_uid, inference_label)

                with open(f"{path}code/prompt.json", 'r') as f:
                    raw_prompt = json.load(f)

                if inference_label == 'modality':
                    prompt = raw_prompt['instruction_modality']
                elif inference_label == 'location':
                    prompt = raw_prompt['instruction_location']
                elif inference_label == 'joint':
                    prompt = raw_prompt['instruction_joint']

                descriptions = description_per_uid[sample['U_id']]
                history = get_block(sample, 'History', 'Case')
                
                for description in descriptions:
                    age = get_block(description, 'Age', 'Description')
                    sex = get_block(description, 'Sex', 'Description')
                    image = Image.open(path+'images/'+description['image']+'.png')
                    vision_query = image_processor(image).unsqueeze(0)

                    if inference_label == 'modality' or inference_label == 'joint':
                        split_text = "The image is a "
                    elif inference_label == 'location':
                        split_text = "The image shows a "
                    
                    text = f"<image>\n{age}{sex}patient.\n{history}\n"

                    vision_x_static_context = torch.cat(static_context_vision + [vision_query], dim=0)
                    vision_x_context = torch.cat(context_vision + [vision_query], dim=0)
                    vision_x = torch.cat([vision_query], dim=0)
                    
                    vision_x_static_context = vision_x_static_context.unsqueeze(1).unsqueeze(0)
                    vision_x_context = vision_x_context.unsqueeze(1).unsqueeze(0)
                    vision_x = vision_x.unsqueeze(1).unsqueeze(0)            
                    
                    lang_x_static_context = tokenizer([prompt+static_context_str+text], return_tensors="pt")
                    lang_x_context = tokenizer([prompt+context_str+text], return_tensors="pt")
                    lang_x = tokenizer([prompt+text], return_tensors="pt")

                    if inference_label == 'modality':
                        label = f"{split_text} {description['Type']} scan."
                    elif inference_label == 'location':
                        label = f"{split_text} {description['Location Category']}."
                    elif inference_label == 'joint':
                        label = f"{split_text} {description['Type']} scan showing a {description['Location Category']}."

                    text_complete = f"{prompt}<image>\n{age}{sex}patient.\n{history}\n{label}<|endofchunk|>{tokenizer.eos_token}"
                    
                    samples.append((vision_x_static_context, vision_x_context, vision_x, lang_x_static_context, lang_x_context, lang_x, tokenizer(label, return_tensors="pt"), tokenizer(text_complete, return_tensors="pt")))

        df_samples = pd.DataFrame(samples, columns=['vision_x_static_context', 'vision_x_context', 'vision_x', 'lang_x_static_context', 'lang_x_context', 'lang_x', 'label', 'text_complete'])

        new_dataset = MyDataset(df_samples['vision_x_static_context'].tolist(), df_samples['vision_x_context'].tolist(), df_samples['vision_x'].tolist(), df_samples['lang_x_static_context'].tolist(), df_samples['lang_x_context'].tolist(), df_samples['lang_x'].tolist(), df_samples['label'].tolist(), df_samples['text_complete'].tolist())

        torch.save(new_dataset, f"{path}data_ft_flamingo/{inference_split}.pt")
                
    print(f"Dataset creation finished")
        
