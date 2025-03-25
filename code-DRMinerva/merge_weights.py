from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
import torch

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default='sapienzanlp/Minerva-3B-base-v1.0')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_dir_model_adapters', type=str, default='dr_minerva_adapters')
    parser.add_argument('--output_dir_model', type=str, default='dr_minerva')

    args = parser.parse_args()
    base_model = args.base_model
    device_name = args.device
    output_dir_model_adapters = args.output_dir_model_adapters
    output_dir_model = args.output_dir_model

    tokenizer = AutoTokenizer.from_pretrained(output_dir_model_adapters)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map={"": device_name},)

    lora_model = PeftModel.from_pretrained(
        base_model,
        output_dir_model_adapters,
        device_map={"": device_name},
        torch_dtype=torch.float16,)

    merged_model = lora_model.merge_and_unload()
    merged_model.save_pretrained(output_dir_model)
    tokenizer.save_pretrained(output_dir_model)

    print('model weights merged and saved in', output_dir_model)