import openai
import json,time



def get_jobscript():
    model = "forge-m-instruct-base1"
    prompt = "generate a Frontier job script for topaz-tft"
    with open('.chathpc_token', 'r') as token:
        TOKEN = token.readline().strip()

    client = openai.OpenAI(
        api_key = f"{TOKEN}",
        base_url = "https://obsidian.ccs.ornl.gov/chathpc/api/v1",
    )

    chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user", "content": f"{prompt}"
                    }
                ],
            model=model,
            temperature=0.7,
            top_p=1,
            max_tokens=512,
            )
    script = chat_completion.choices[0].message.content
    return script 


