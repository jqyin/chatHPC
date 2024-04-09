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
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast

#model_path = "/lustre/orion/proj-shared/stf218/junqi/meta-llama/Llama-2-7b-chat-hf"
model_path = "/lustre/orion/proj-shared/stf218/junqi/chathpc/forge-s-instruct-base1"
model_kwargs = {'device':'cuda'}
encode_kwargs = {'normalize_embeddings': True}

embeddings = HuggingFaceEmbeddings(
    model_name=model_path,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

if "llama" not in model_path:
    embeddings.client.tokenizer = GPTNeoXTokenizerFast.from_pretrained(model_path, use_fast=True)
embeddings.client.tokenizer.pad_token = embeddings.client.tokenizer.eos_token

texts = [

    #"""scheduling policy of frontier user guide is that In a simple batch queue system, jobs run in a first-in, first-out (FIFO) order. This can lead to inefficient use of the system. If a large job is the next to run, a strict FIFO queue can cause nodes to sit idle while waiting for the large job to start. Backfilling would allow smaller, shorter jobs to use those resources that would otherwise remain idle until the large job starts. With the proper algorithm, they would do so without impacting the start time of the large job. While this does make more efficient use of the system, it encourages the submission of smaller jobs""",
    #"""Frontier is a HPE Cray EX supercomputer located at the Oak Ridge Leadership Computing Facility. With a theoretical peak double-precision performance of approximately 2 exaflops (2 quintillion calculations per second), it is the fastest system in the world for a wide range of traditional computational science applications. The system has 74 Olympus rack HPE cabinets, each with 128 AMD compute nodes, and a total of 9,408 AMD compute nodes.""",
    #"""Each Frontier compute node consists of [1x] 64-core AMD “Optimized 3rd Gen EPYC” CPU (with 2 hardware threads per physical core) with access to 512 GB of DDR4 memory. Each node also contains [4x] AMD MI250X, each with 2 Graphics Compute Dies (GCDs) for a total of 8 GCDs per node. The programmer can think of the 8 GCDs as 8 separate GPUs, each having 64 GB of high-bandwidth memory (HBM2E). The CPU is connected to each GCD via Infinity Fabric CPU-GPU, allowing a peak host-to-device (H2D) and device-to-host (D2H) bandwidth of 36+36 GB/s. The 2 GCDs on the same MI250X are connected with Infinity Fabric GPU-GPU with a peak bandwidth of 200 GB/s. The GCDs on different MI250X are connected with Infinity Fabric GPU-GPU in the arrangement shown in the Frontier Node Diagram below, where the peak bandwidth ranges from 50-100 GB/s based on the number of Infinity Fabric connections between individual GCDs.""",
    #"""file system of frontier user guide is that Frontier is connected to Orion, a parallel filesystem based on Lustre and HPE ClusterStor, with a 679 PB usable namespace (/lustre/orion/). In addition to Frontier, Orion is available on the OLCF’s data transfer nodes. It is not available from Summit. Data will not be automatically transferred from Alpine to Orion. Frontier also has access to the center-wide NFS-based filesystem (which provides user and project home areas). Each compute node has two 1.92TB Non-Volatile Memory storage devices. See Data and Storage for more information.""", 
    #"""system interconnect of frontier is that the Frontier nodes are connected with [4x] HPE Slingshot 200 Gbps (25 GB/s) NICs providing a node-injection bandwidth of 800 Gbps (100 GB/s).""",

    """scheduling policy of frontier is that in a simple batch queue system, jobs run in a first-in, first-out (FIFO) order.""",
    """Frontier is a HPE Cray EX supercomputer located at the Oak Ridge Leadership Computing Facility. """,
    """Each Frontier compute node consists of [1x] 64-core AMD “Optimized 3rd Gen EPYC” CPU (with 2 hardware threads per physical core) with access to 512 GB of DDR4 memory. Each node also contains [4x] AMD MI250X, each with 2 Graphics Compute Dies (GCDs) for a total of 8 GCDs per node.""",
    """system interconnect of frontier is that the Frontier nodes are connected with [4x] HPE Slingshot 200 Gbps (25 GB/s) NICs providing a node-injection bandwidth of 800 Gbps (100 GB/s).""",
    """File systems of frontier is that Frontier is connected to Orion, a parallel filesystem based on Lustre and HPE ClusterStor, with a 679 PB usable namespace (/lustre/orion/).""",
]

smalldb = Chroma.from_texts(texts, embedding=embeddings)
question = "what is the cpu type on frontier"
#docs = smalldb.max_marginal_relevance_search(question,k=1, fetch_k=3)
docs = smalldb.similarity_search(question,k=3)

print(docs)


llama_cpp_path = "/lustre/orion/proj-shared/stf218/junqi/llama-cpp/llama-2-7b-chat.gguf.q5_1.bin"
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
n_gpu_layers = 40 
n_batch = 512

llm = LlamaCpp(
 model_path=llama_cpp_path,
 max_tokens=512,
 n_ctx = 4096,
 temperature = 0.1,
 n_gpu_layers=n_gpu_layers,
 n_batch=n_batch,
 top_p=1.0,
 repeat_penalty=1.2,
 top_k=50,
 callback_manager=callback_manager,
 verbose=True)


template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)# Run chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=smalldb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)


result = qa_chain({"query": question})
print(result)



