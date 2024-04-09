from selfcheckgpt.modeling_selfcheck import SelfCheckNLI
import torch
import pandas as pd
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
##Instruction: The File should be CSV, there should be 11 columns -> Question, Sample 1, Sample 2, Sample 3...Sample 11. 
## With Sample 1 being the main answer
## Also please make sure, the main answer i.e sample 1 is not in points, like doesn't have numbered points or bullet points. Paras would be better.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
selfcheck_nli = SelfCheckNLI(device=device) # set device to 'cuda' if GPU is available


def calc_nli_score(df, number_of_samples):
    main_answer = []
    sentences = []
    main_sentences = []
    scores = []
    samplepass = []
    for i in range(len(df)):
        main_answer.append(df['chosen'][i])
    for i in range(len(main_answer)):
        sentences.append(sent_tokenize(main_answer[i]))
    for i in range(len(sentences)):
        ans = ""
        for idx, sentence in enumerate(sentences[i], start=1):
            ans+= f"{idx}. {sentence}"
        main_sentences.append(ans)
    for i in range(len(df)):
        df['chosen'][i] = main_sentences[i]
    for i in range(len(df)):
        samples = []
        for j in range(number_of_samples):
            samples.append(df[f"chathpc{j+1}"][i])
        samplepass.append(samples)
    for i in range(len(sentences)):
        score = selfcheck_nli.predict(
            sentences = sentences[i],                          # list of sentences
            sampled_passages = samplepass[i] # list of sampled passages
        )
        print(score)
        l1= np.linalg.norm(score, ord=1)
        scores.append(l1)
    df['Scores'] = scores
    return df
