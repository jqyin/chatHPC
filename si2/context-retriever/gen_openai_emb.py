from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, JSONLoader
import pandas as pd

file_path = "../../data/train/model/"

loader = DirectoryLoader(file_path,
                         glob='**/*.jsonl',
                         show_progress=True,
                         loader_cls=JSONLoader,
                         loader_kwargs={'jq_schema':'.text', 'text_content':False, 'json_lines':True}
                        )

docs = loader.load()
print(len(docs))


text_splitter = RecursiveCharacterTextSplitter(
    #separators=["\n\n", "\n", " "],
    chunk_size=600,
    chunk_overlap=50,
    length_function=len
)

splits = text_splitter.split_documents(docs)

data = {
    'olcf-doc':[doc.page_content for doc in splits],
}

df = pd.DataFrame(data)
print(df)

client = OpenAI()

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

df['ada_embedding'] = df['olcf-doc'].apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))
df.to_csv('output/embedded_olcf_docs.csv', index=False)
