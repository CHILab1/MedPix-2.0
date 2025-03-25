from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm
import argparse
import json

from utils import get_block, load_complete_dataset, load_complete_dataset_description

def load_retriever(retriever_path, emb_model_name):
    print('loading retriever...')
    embeddings_model = HuggingFaceEmbeddings(
        model_name=emb_model_name,
        #multi_process=True,
        model_kwargs={"device": "cuda:0", "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True}) # Set `True` for cosine similarity
    
    vectorstore = FAISS.load_local(retriever_path, embeddings_model, allow_dangerous_deserialization = True)

    retriever = vectorstore.as_retriever()
    print('retriever loaded.')
    return retriever

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train_2')
    args = parser.parse_args()
    split = args.split

    corpus_rag = 'history'

    # embedding model definition
    emb_model = 'mistral'
    emb_model_name = "Linq-AI-Research/Linq-Embed-Mistral"
    emb_model_str = 'linq-embed-mistral'

    dataset = load_complete_dataset()
    path = "MedPix-2-0/"

    # retriever model definition
    retriever = load_retriever(path+"MedPix/vectorstore-"+corpus_rag+"-"+emb_model_str, emb_model_name)

    retrieved_reports = {}

    # clustering descriptions per uid
    dataset_description = load_complete_dataset_description()
    description_per_uid = {}
    for sample in dataset_description[split]:
        if sample['U_id'] not in description_per_uid:
            description_per_uid[sample['U_id']] = []
        description_per_uid[sample['U_id']].append(sample)

    for idx, sample in enumerate(tqdm(dataset[split])):
        if corpus_rag == 'history':
            descriptions = description_per_uid[sample['U_id']]
            age = get_block(descriptions[0], 'Age', 'Description')
            sex = get_block(descriptions[0], 'Sex', 'Description')
            history = get_block(sample, 'History', 'Case')
            query = f'{age}{sex}patient.\n{history}'
            retrieved_data = retriever.invoke(query)
            retrieved_reports[sample['U_id']] = [i.metadata['U_id'] for i in retrieved_data]
    
    with open(path+"retrieved-over-"+split+"-"+corpus_rag+"-"+emb_model_str+".json", 'w') as f:
        json.dump(retrieved_reports, f)

    print(f'retrieved reports for {split} over {corpus_rag} w/ {emb_model} saved.') 