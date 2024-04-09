import openai
import json,time

model = "forge-m-instruct-base1"
eval_data = "eval.txt"
trial_name = 'chathpc'
output_data = f"eval_{trial_name}.txt"

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
            



