from rouge_score import rouge_scorer
import pandas as pd
import numpy as np
from bert_score import score
from sentence_transformers import SentenceTransformer, util
##The csv file should have 3 columns -> 'queries', 'x' (this should be ground_truth), 'y' (generated answers)
bart_model = SentenceTransformer('facebook/bart-large-cnn')

def calc_bertscore(y_list, x_list, df, lang="en", model_type="bert-large-uncased"):
    P, R, F1 = score(y_list, x_list, lang=lang, verbose=True, model_type=model_type)
    df['Bert F1 Score'] = F1
    df['Bert Precision'] = P
    df['Bert Recall'] = R

    return df

def calc_rougescore(y,x,df):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    # x = df['x'].to_list()
    # y = df['y'].to_list()
    z = []
    for i in range(len(df)):
        scores = scorer.score(y[i], x[i])
        z.append(scores)
    rouge1p = []
    rougeLp = []
    rouge1r = []
    rougeLr = []
    rouge1f1 = []
    rougeLf1 = []
    for i in range(len(df)):
        rouge1p.append(z[i]['rouge1'].precision)
        rouge1r.append(z[i]['rouge1'].recall)
        rouge1f1.append(z[i]['rouge1'].fmeasure)
        rougeLp.append(z[i]['rougeL'].precision)
        rougeLr.append(z[i]['rougeL'].recall)
        rougeLf1.append(z[i]['rougeL'].fmeasure)
    df['Rouge 1 Precision'] = rouge1p
    df['Rouge 1 Recall'] = rouge1r
    df['Rouge 1 F1 Score'] = rouge1f1
    df['Rouge L Precision'] = rougeLp
    df['Rouge L Recall'] = rougeLr
    df['Rouge L F1 Score'] = rougeLf1

    return df

def calc_bartscore(y,x,df):
    bartres = []
    for i in range(len(df)):
        embedding1 = bart_model.encode(y[i], convert_to_tensor = True)
        embedding2 = bart_model.encode(x[i], convert_to_tensor = True)
        similarity_score = util.pytorch_cos_sim(embedding1, embedding2)
        bartres.append(similarity_score)
    bartres = [tensor.item() for tensor in bartres]
    df['Bart Score'] = bartres
    return df