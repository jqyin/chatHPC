from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import UnstructuredRSTLoader
from langchain.vectorstores.utils import filter_complex_metadata
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast
import pandas as pd
import csv

top=5
#model_path = "/lustre/orion/proj-shared/stf218/junqi/embedder/model/meta-llama_Llama-2-7b-hf"
#model_path = "model/UAE-finetuned"
model_path = "WhereIsAI/UAE-Large-V1"
#model_path = "/lustre/orion/proj-shared/stf218/junqi/chathpc/forge-s-instruct-base1"

model_kwargs = {'device':'cuda'}
encode_kwargs = {'normalize_embeddings': True}
persist_directory = '../../../emb_db_store'

#scoring prompt for chatgpt: "For the question "how many nodes are there on Frontier", re-order the following 10 page_content based on the relevance to the question. The expected answer should be a ranking list of 10 numbers."

embeddings = HuggingFaceEmbeddings(
    model_name=model_path,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

if "forge" in model_path:
    embeddings.client.tokenizer = GPTNeoXTokenizerFast.from_pretrained(model_path, use_fast=True)
    embeddings.client.tokenizer.pad_token = embeddings.client.tokenizer.eos_token
elif "llama" in model_path:
    embeddings.client.tokenizer.pad_token = embeddings.client.tokenizer.eos_token



db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)
print(db._collection.count())


ground_truth = pd.read_csv("../data/eval/retriever/ground-truth-top1.csv", header=0)

hit = 0
for index, record in ground_truth.iterrows(): 
    question = record['prompt']
    answer = record['doc']
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", question)
    docs = db.similarity_search(question, k=top)
    docs = [doc.page_content  for doc in docs]
    print(docs)
    if answer in docs:
        hit = hit + 1 
        print("hit")

acc = 1.0*hit/len(ground_truth)
print(f"{model_path} top{top} acc: {acc}")





