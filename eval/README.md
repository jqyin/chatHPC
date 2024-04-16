# Evaluation
Our evaluation is based on [generic language benchmark](https://github.com/EleutherAI/lm-evaluation-harness), [lexical](https://github.com/google-research/google-research/tree/master/rouge), [semantic](https://github.com/Tiiiger/bert_score), [hallucination](https://github.com/potsawee/selfcheckgpt), and [privacy](https://github.com/AI-secure/DecodingTrust).  

Our aggregated metric is calculated using [this script](./model/get_aggregated_metric.py). 

All the evaluation logs can be found [here](https://www.dropbox.com/scl/fo/6tfniwof3s4tshh16wj5h/APK9KfJXhxq6iMnukWoeJ60?rlkey=98xwip2fqxkepwqvddpk2pwhk&dl=0)

## Model 
The evaluation is [streamlined](./model/chathpc_eval.py) via chatHPC API,
```python
import openai
with open('.chathpc_token', 'r') as token:
        TOKEN = token.readline().strip()

client = openai.OpenAI(
        api_key = f"{TOKEN}",
        base_url = "https://obsidian.ccs.ornl.gov/chathpc/api/v1",
    )

with open(eval_data, 'r') as input_file:
        with open(output_data, 'a' ) as output_file:
            for line in input_file:
                data = json.loads(line)
                print(data["prompt"])
                chat_completion = client.chat.completions.create(
                        messages=[
                            {
                                "role": "user", "content": f"{data['prompt']}"
                                }
                            ],
                        model=model,
                        temperature=0.7,
                        top_p=1,
                        max_tokens=512,
                        )
                data[f'{trial_name}'] = chat_completion.choices[0].message.content
                output_file.write(json.dumps(data) + '\n')
```
The [lexical](https://github.com/google-research/google-research/tree/master/rouge) and [semantic](https://github.com/Tiiiger/bert_score) metrics are evaluated against the [ground truth](https://github.com/jqyin/chatHPC/blob/main/data/val/model/eval_ground_truth.txt). 

Lexical, Semantic, and Hallucination metrics are then calculated following [3 steps](./model/run.sh):
```bash
#setp 1: generate 5 responses via chatHPC API 
python -u chathpc_eval.py --eval_data ../data/val/model/eval_ground_truth.txt --trial_name chathpc1

for trial in $(seq 1 4)
do
   python -u chathpc_eval.py --eval_data eval_chathpc${trial}.txt --trial_name chathpc$((trial+1))
done

#step 2: calculate lexical, semantic, and hallucination scores 
python score.py eval_chathpc5.txt

#step 3: calculate our metric 
python get_aggregated_metric.py eval_chathpc5-scores.csv
```

The privacy evaluation is part of [SI^2](https://github.com/jqyin/chatHPC/tree/main/si2/README.md)

## Retriever
The retrieval accuracy of the embedding model is evaluated using [this script](./retriever/eval_embedder.py) against the [ground truth](https://github.com/jqyin/chatHPC/blob/main/data/val/retriever/ground-truth-top1.csv) generated from OpenAI-ada-002 embedding model. 

## Continual Learning (CL)
We use [generic language benchmark](https://github.com/EleutherAI/lm-evaluation-harness) to evaluate model performance after continual learning. 
```bash
lm_eval --model hf --model_args pretrained=forge-m2-olcfdoc-200s-olcfdocv4/ --tasks arc_easy,arc_challenge,sciq,piqa,openbookqa  --device cuda:0 --batch_size 32
```
The [run script](./cl/run.sh) is provided, along with evaluation [logs](https://github.com/jqyin/chatHPC/blob/main/eval/cl/eval.forge-cl-tft). 



