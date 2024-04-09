from openai import OpenAI
import pandas as pd
import numpy as np
import csv, ast

top_n  = 5
client = OpenAI()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def search_reviews(df, question, n=3, pprint=True):
   embedding = get_embedding(question)
   
   df['similarities'] = df.ada_embedding.apply(lambda x: cosine_similarity(ast.literal_eval(x), embedding))
   res = df.sort_values('similarities', ascending=False).head(n)
   return res

df = pd.read_csv("output/embedded_olcf_docs.csv")

with open("sampled_prompts1.txt", 'r') as file:
    questions = file.readlines()

with open("sample-retrival-openai.csv", 'w', newline='', encoding='utf-8') as file: 
    writer = csv.writer(file, quoting=csv.QUOTE_ALL)
    writer.writerow(['text1', 'text2', 'label'])
    for question in questions: 
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", question)
        res = search_reviews(df, question, n=top_n)
        for _, row in res.iterrows():
            writer.writerow([question, row['olcf-doc'], row['similarities']*top_n])




