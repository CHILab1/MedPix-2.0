from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from tqdm import tqdm
import json
import os

from utils import get_block, load_complete_dataset, load_complete_dataset_description

def load_corpus_per_RAG(path, metadata):
    docs = os.listdir(path)
    corpus = []
    for doc in tqdm(docs):
        if doc.endswith('.txt'):
            with open(path+doc, 'r') as f:
                doc_text = f.readlines()
            doc_text_parsed = ''.join(doc_text)
            corpus.append(Document(page_content=doc_text_parsed, metadata=metadata[doc[:-4]]))
    return corpus

if __name__ == '__main__':

    path = 'MedPix-2-0/'
    dataset = load_complete_dataset()
    dataset_description = load_complete_dataset_description()

    rag_path = path + "rag-corpus-history/"
    os.makedirs(rag_path, exist_ok=True)

    # clustering descriptions per uid
    description_per_uid = {}
    for sample in dataset_description['train_1']:
        if sample['U_id'] not in description_per_uid:
            description_per_uid[sample['U_id']] = []
        description_per_uid[sample['U_id']].append(sample)

    # generate RAG corpus - one document per u_id
    # store metadata of the document
        
    metadata_documents = {}

    for sample in tqdm(dataset['train_1']):
        uid = sample['U_id']
        descriptions = description_per_uid[uid]

        # for generic info pick the first description
        age = get_block(descriptions[0], 'Age', 'Description')
        sex = get_block(descriptions[0], 'Sex', 'Description')
        case_diagnosis = get_block(sample, 'Case Diagnosis', 'Case')
        history = get_block(sample, 'History', 'Case')

        document = f"{age}{sex}patient suffering from a {case_diagnosis}.\n{history}"

        metadata_documents[sample['U_id']] = {
            'U_id': sample['U_id'],
            'case_diagnosis': case_diagnosis[:-1]}

        with open(rag_path + f"{sample['U_id']}.txt", 'w') as f:
            f.write(document)

    with open(rag_path + "metadata.json", 'w') as f:
        json.dump(metadata_documents, f)

    emb_model = 'mistral'
    model_name = "Linq-AI-Research/Linq-Embed-Mistral"
    model_str = 'linq-embed-mistral'

    print("Loading data...")

    with open(rag_path + "metadata.json", 'r') as f:
        metadata = json.load(f)

    corpus = load_corpus_per_RAG(rag_path, metadata)

    print("Splitting text...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    
    all_splits = text_splitter.split_documents(corpus)

    print("Loading embeddings...")

    embeddings_model = HuggingFaceEmbeddings(
        model_name=model_name,
        #multi_process=True,
        model_kwargs={"device": "cuda:0", "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True}) # Set `True` for cosine similarity

    print("Creating vector store...")

    vectorstore = FAISS.from_documents(documents=all_splits, embedding=embeddings_model)

    vectorstore.save_local(path+"MedPix/vectorstore-history-"+model_str)