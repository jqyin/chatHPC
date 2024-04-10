import openai, re, os, glob

openai.api_key = os.getenv("OPENAI_API_KEY")

def table_to_text(table, context):
    prompt = f"Generate a paragraph providing a detailed description of the content presented in the following table. This table pertains to {context}. Append the table in markdown format at the end for reference. \n{table}"
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=2048, 
        temperature=0.1,
        top_p=1 
    )
    return response['choices'][0]['text'].strip()

file_paths = glob.glob("*_tables.txt")
for file_path in file_paths:
    with open(file_path, "r") as file:
        file_contents = file.read()

    tables = re.split(r"Table \d+:", file_contents.strip())

    basename = os.path.splitext(os.path.basename(file_path))[0]
    output_file_path = basename + "_texts.txt"
    context = re.split(r'[_-]', basename) 
    context = " ".join(context[:-1:])
    
    if not os.path.exists(output_file_path):
        with open(output_file_path, "w") as output_file:
            for i, table in enumerate(tables):
                if table.strip():  # Ensure there is content in the table
                    generated_text = table_to_text(table, context)
                    print(f"{file_path} table:{i} \n {generated_text}")
                    output_file.write(f"Table {i}:\n{generated_text}\n\n")


