
from typing import List, Optional
import fire, glob, os
from llama import Llama, Dialog
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import UnstructuredRSTLoader
from langchain.vectorstores.utils import filter_complex_metadata

context = "Frontier"

model_path = "/lustre/orion/scratch/junqi/stf218/chathpc/chatHPC/retrival/save/finetuned-embedder"
model_kwargs = {'device':'cuda'}
encode_kwargs = {'normalize_embeddings': True}
persist_directory = '/lustre/orion/scratch/junqi/stf218/chathpc/chatHPC/retrival/emb_db_store'
embeddings = HuggingFaceEmbeddings(
    model_name=model_path,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)

def concatenate_docs(documents):
    total_word_count = 0
    result = []

    for doc in documents:
        words = doc.page_content.split()

        # Check if adding this document's content will exceed the word limit
        if total_word_count + len(words) <= 500:
            result.append(' '.join(words))
            total_word_count += len(words)
        else:
            # Find where to cut off to stay within the word limit
            remaining_words = 500 - total_word_count
            remaining_text = ' '.join(words[:remaining_words])

            # Append the truncated text and stop
            result.append(remaining_text)
            break
    return '\n'.join(result)



def generate(
    context_question,
    ticket_data,
    output_file,
    generator,
    nquestion,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_gen_len: Optional[int] = None,
):

    context_question = f"This is about {context}. "+context_question 
    docs = db.similarity_search_with_score(context_question, k=1)
    print(docs)
    #docs  = concatenate_docs(docs)
    if docs[0][1] < 0.6:  # related 
        docs = "For your consideration, following is some additional information related to your question:\n"+docs[0][0].page_content
    else:
        docs = ''
    dialogs: List[Dialog] = [
            [
{"role": "system", "content": "you are a helpful user assistant, skilled in high performance computing."},
{"role": "user", "content": f"""
Use the following email exchanges about {context} as input to generate {nquestion} instruction (Q:) and response (A:) pairs. Requirements: 1. Remove all the email headers, footers, and greetings. 2. Do not include personal information such as id, uid, username, account, name, and email address. 3. The instruction should be diverse and include both questions and imperative instructions. 4. Be generic in the instruction and response, but keep the details and related context ({context}) in the insturction. 

Input:    
{ticket_data}
{docs}
"""
    }],
    ]
    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        output_file.write(
            f"{result['generation']['content']}\n\n"
        )
        #print("\n==================================\n")
     


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    ticket_path: str,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_seq_len: int = 4096,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    
    tickets = glob.glob(ticket_path+"/*.csv")
    for ticket in tickets: 
        df = pd.read_csv(ticket, header=0)
        if len(df) > 0: 
            try:
                basename = os.path.splitext(os.path.basename(ticket))[0]
                filename = basename + "_parsed.txt"
                with open(filename, 'w', encoding="utf-8") as output_file:
                    ticket_data = ' '.join(df['prompt'] + ' ' + df['response'])
                    print(ticket, len(ticket_data))
                    nquestion = len(df)
                    if len(ticket_data) > max_seq_len*2/3.0:
                        for idx, row in df.iterrows():
                            ticket_data = row['prompt'] + ' ' + row['response']
                            print(ticket, idx, len(ticket_data)) 
                            if len(ticket_data) > max_seq_len*2/3.0:
                                continue
                            nquestion = 1
                            generate(df['prompt'][0], ticket_data, output_file, generator, nquestion)   
                    else:
                        generate(df['prompt'][0],ticket_data, output_file, generator, nquestion)   
            except:
                pass     


if __name__ == "__main__":
    fire.Fire(main)
