from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import UnstructuredRSTLoader
from langchain.vectorstores.utils import filter_complex_metadata
from langchain.llms import LlamaCpp
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chains import RetrievalQA
from transformers import AutoConfig, OPTForCausalLM, GPTNeoXForCausalLM, AutoTokenizer, GPTNeoXTokenizerFast
from transformers import pipeline, set_seed
import os, re

question = "how do i get access to quantum computing resources via OLCF?"
#model_path = "meta-llama/Llama-2-7b-chat-hf"
emb_path = "WhereIsAI/UAE-Large-V1"
#emb_path = "/lustre/orion/proj-shared/stf218/junqi/chathpc/forge-s-instruct-base1"
model_path = "/lustre/orion/proj-shared/stf218/junqi/chathpc/forge-m-all-60k"
#model_path = "/lustre/orion/proj-shared/stf218/junqi/chathpc/forge-m-instruct-base1"
model_kwargs = {'device':'cuda'}
encode_kwargs = {'normalize_embeddings': True}
persist_directory = './emb_db_store'

llama_cpp_path = "/lustre/orion/scratch/junqi/stf218/chathpc/chatHPC/retrival/.cache/huggingface/hub/models--TheBloke--Llama-2-7B-chat-GGML/snapshots/76cd63c351ae389e1d4b91cab2cf470aab11864b/llama-2-7b-chat.gguf.q5_1.bin"

embeddings = HuggingFaceEmbeddings(
    model_name=emb_path,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
if "forge" in emb_path:
    embeddings.client.tokenizer = GPTNeoXTokenizerFast.from_pretrained(emb_path, use_fast=True)
    embeddings.client.tokenizer.pad_token = embeddings.client.tokenizer.eos_token
elif "llama" in emb_path:
    embeddings.client.tokenizer.pad_token = embeddings.client.tokenizer.eos_token

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)
print(db._collection.count())


docs = db.similarity_search(question, k=5)
print(docs)
content = [doc.page_content for doc in docs]
strings = ' '.join(content)
words = strings.split()
inputs = ' '.join(words[:1500])

#Keep the answer as concise as possible.  
#prompt = f"""Answer the question at the end. Use following input as additional context if it is related, and skip it if it is not. 
prompt = f"""Answer the question at the end. Use following additional input if the answer is related, and ignore the input if not.

Input:
{inputs}

Human: {question}
Assistant:"""
print(prompt)

#input:
#- {docs[0].page_content}
#- {docs[1].page_content}
#- {docs[2].page_content}
#{docs[3].page_content}
#{docs[4].page_content}"

def process_response(response, num_rounds):
    output = str(response[0]["generated_text"])
    output = output.replace("<|endoftext|></s>", "")
    all_positions = [m.start() for m in re.finditer("Human: ", output)]
    place_of_second_q = -1
    if len(all_positions) > num_rounds:
        place_of_second_q = all_positions[num_rounds]
    if place_of_second_q != -1:
        output = output[0:place_of_second_q]
    return output


def get_generator(path):
    if os.path.exists(path):
        # Locally tokenizer loading has some issue, so we need to force download
        model_json = os.path.join(path, "config.json")
        if os.path.exists(model_json):
            #model_json_file = json.load(open(model_json))
            #model_name = model_json_file["_name_or_path"]
            tokenizer = GPTNeoXTokenizerFast.from_pretrained(path,
                                                      fast_tokenizer=True)
    else:
        tokenizer = GPTNeoXTokenizerFast.from_pretrained(path, fast_tokenizer=True)

    #tokenizer.pad_token = tokenizer.eos_token

    model_config = AutoConfig.from_pretrained(path)
    model = GPTNeoXForCausalLM.from_pretrained(path,
                                           from_tf=bool(".ckpt" in path),
                                           config=model_config).half()

    print("number of paramters: ", sum(p.numel() for p in model.parameters()))
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    generator = pipeline("text-generation",
                         model=model,
                         tokenizer=tokenizer,
                         device="cuda:0")
    return generator


generator = get_generator(model_path)

response = generator(prompt,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.7,
                        top_p=1.,
                        )

output = process_response(response, 1)

print(output)
