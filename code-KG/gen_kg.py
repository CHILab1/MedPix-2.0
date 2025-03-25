from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import KnowledgeGraphIndex
from llama_index.core import StorageContext
from llama_index.core import Settings
from llama_index.core import Document

from tqdm import tqdm
import pandas as pd
import argparse
import torch

from pyvis.network import Network
import networkx as nx
import matplotlib.pyplot as plt
import pickle

def plot_graph(graph):
    print(graph)
    plt.figure(figsize=(25, 25))
    pos = nx.circular_layout(graph)
    nx.draw(graph, with_labels=True, node_color='skyblue', node_size=150, pos=pos)
    nx.draw_networkx_edge_labels(graph, pos=pos)
    if lower:
        plt.savefig(f"{path}KG/graphs-{split}-{relations}trips{llm_model}-lower.png", format="PNG")
    else:
        plt.savefig(f"{path}KG/graphs-{split}-{relations}trips{llm_model}.png", format="PNG")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_model', type=str, default='llama31inst')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--relations', type=int, default=3)

    parser.add_argument('--lower', action='store_true')
    parser.add_argument('--no-lower', dest='lower', action='store_false')
    parser.set_defaults(lower=False)

    args = parser.parse_args()
    llm_model = args.llm_model
    device = args.device
    split = args.split
    relations = args.relations
    lower = args.lower

    device = device if torch.cuda.is_available() else "cpu"
    
    if llm_model == 'llama31inst':
        # path to local llama 3.1 8B inst
        llm_model_path = 'LLM/llama31inst/'

    path = 'MedPix-2-0/'

    # From created docs templates
    df = pd.read_csv(f'{path}KG/templates/template-{split}.csv')
    documents = []
    for i in tqdm(range(len(df)), desc='loading documents'):
        if lower:
            documents.append(Document(text=df.at[i, 'Description'].lower(), metadata={"U_id": df.at[i, 'U_id'].lower(), "Descriptions": df.at[i, 'Description'].lower()}))
        else:
            documents.append(Document(text=df.at[i, 'Description'], metadata={"U_id": df.at[i, 'U_id'], "Descriptions": df.at[i, 'Description']}))

    print('loading llm ...')
    Settings.llm = HuggingFaceLLM(model_name=llm_model_path,
                                    tokenizer_name=llm_model_path,
                                    device_map=device,)

    Settings.embed_model = None
    Settings.chunk_size = 4096
    
    #setup the storage context
    graph_store = SimpleGraphStore()
    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    print(f'creating knowledge graph {split}')

    #Construct the Knowlege Graph Index
    index = KnowledgeGraphIndex.from_documents(documents=documents,
                                            max_triplets_per_chunk=relations,
                                            #max_length=max_tokens,
                                            show_progress=True,
                                            storage_context=storage_context,
                                            )

    if lower:
        index.storage_context.persist(persist_dir=f"{path}KG/graphs/{split}-{relations}trips{llm_model}-lower")
    else:
        index.storage_context.persist(persist_dir=f"{path}KG/graphs/{split}-{relations}trips{llm_model}")

    ## create graph for visualization

    g = index.get_networkx_graph()

    plot_graph(g)
    if lower:
        with open(f"{path}KG/graphs-{split}-{relations}trips{llm_model}-lower.pickle", 'wb') as f:
            pickle.dump(g,f)
    else:
        with open(f"{path}KG/graphs-{split}-{relations}trips{llm_model}.pickle", 'wb') as f:
            pickle.dump(g,f)

    net = Network(notebook=True, cdn_resources="in_line", directed=True)
    net.from_nx(g)
    if lower:
        net.show(f"{path}KG/graphs-{split}-{relations}trips{llm_model}-lower.html")
    else:
        net.show(f"{path}KG/graphs-{split}-{relations}trips{llm_model}.html")