from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import UnstructuredRSTLoader
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.vectorstores.utils import filter_complex_metadata
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, JSONLoader, UnstructuredMarkdownLoader
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast

#model_path = "../save/finetuned-embedder/"
#model_path = "/lustre/orion/proj-shared/stf218/junqi/embedder/model/meta-llama_Llama-2-7b-hf"
model_path = "WhereIsAI/UAE-Large-V1"
#model_path = "/lustre/orion/proj-shared/stf218/junqi/chathpc/forge-s-instruct-base1"
model_kwargs = {'device':'cuda'}
encode_kwargs = {'normalize_embeddings': True}
persist_directory = './emb_db_store'

olcf_file_path = "../data/train/model/"

loader_olcf = DirectoryLoader(olcf_file_path,
                         glob='**/*.jsonl',
                         show_progress=True,
                         loader_cls=JSONLoader,
                         loader_kwargs={'jq_schema':'.text', 'text_content':False, 'json_lines':True}
                        )

docs_olcf = loader_olcf.load()
print("doc-olcf:",len(docs_olcf))

alcf_file_path = "/lustre/orion/scratch/junqi/stf218/chathpc/chatHPC/data/raw/user-guides/"
loader_alcf = DirectoryLoader(alcf_file_path,
                         glob='**/*.md',
                         show_progress=True,
                         loader_cls=UnstructuredMarkdownLoader,
                         loader_kwargs={'mode':'single', 'strategy':'fast'}
                        )

docs_alcf = loader_alcf.load()
print("doc-alcf:",len(docs_alcf))

nersc_file_path = "/lustre/orion/scratch/junqi/stf218/chathpc/chatHPC/data/raw/nersc.gitlab.io/"
loader_nersc = DirectoryLoader(nersc_file_path,
                         glob='**/*.md',
                         show_progress=True,
                         loader_cls=UnstructuredMarkdownLoader,
                         loader_kwargs={'mode':'single', 'strategy':'fast'}
                        )
docs_nersc = loader_nersc.load()
print("doc-nersc",len(docs_nersc))

docs = docs_olcf + docs_alcf + docs_nersc

text_splitter = RecursiveCharacterTextSplitter(
    #separators=["\n\n", "\n", " "],
    chunk_size=600,
    chunk_overlap=50,
    length_function=len
)

splits = text_splitter.split_documents(docs)
print(len(splits))
print(splits) 

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

db = Chroma.from_documents(
    documents=splits, 
    embedding=embeddings,
    persist_directory=persist_directory
)
print(db._collection.count())


