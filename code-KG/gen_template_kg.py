from tqdm import tqdm
import pandas as pd
import argparse
import os

from utils import get_block, get_description_per_uid, load_complete_dataset, load_complete_dataset_description, load_model

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--inference_split', type=str, default='train')
    args = parser.parse_args()
    inference_split = args.inference_split

    path = 'MedPix-2-0/'

    dataset = load_complete_dataset()
    dataset_description = load_complete_dataset_description()

    # clustering descriptions per uid
    description_per_uid = get_description_per_uid(dataset_description)

    os.makedirs(path+'/KG/templates/', exist_ok=True)

    samples = []

    description_string = "{u_id} is a clinical report of a {age}{sex} patient suffering from a {title_specific} displayed in {modality}.{history} The disease {title_specific} located in {location} ({location_category}){treatment_followup}. About {title_generic} we can say that: {disease_discussion}."

    for idx, sample in enumerate(tqdm(dataset[inference_split])):
        u_id = sample['U_id']
        descriptions = description_per_uid[u_id]
        disease_discussion = get_block(sample, 'Disease Discussion', 'Topic')
        treatment_followup = get_block(sample, 'Treatment & Follow Up', 'Case')
        history = get_block(sample, 'History', 'Case')
        title_specific = get_block(sample, 'Title', 'Case')
        title_generic = get_block(sample, 'Title', 'Topic')
        
        if treatment_followup != '':
            treatment_followup =f' and it can be treated in this way: {treatment_followup}'
        
        if history != '':
            history = ' ' + history
            if history[-2:] != '. ':
                history += '.'
            else:
                history = history[:-1]

        for description in descriptions:
            age = get_block(description, 'Age', 'Description')
            sex = get_block(description, 'Sex', 'Description')
            modality = get_block(description, 'Modality', 'Description')
            location = description['Location'] 
            location_category = description['Location Category'] 

            if sex != '':
                sex = ' '+sex

            samples.append((u_id, description_string.format(u_id=u_id, age=age, sex=sex, title_specific=title_specific, modality=modality, history=history, location=location, location_category=location_category, treatment_followup=treatment_followup, title_generic=title_generic, disease_discussion=disease_discussion)))

    df = pd.DataFrame(samples, columns=['U_id', 'Description'])

    df.to_csv(path+'KG/templates/template-'+inference_split+'.csv', index=False)