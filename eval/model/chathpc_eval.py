import openai
import json
import time
import argparse

def main(eval_data, trial_name):
    model = "llama-2-13b-chat-hf"
    #model = "forge-m-instruct-base1"
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatHPC Evaluation")
    parser.add_argument("--eval_data", help="Path to the evaluation data file")
    parser.add_argument("--trial_name", help="Name of the trial")
    args = parser.parse_args()

    main(args.eval_data, args.trial_name)

