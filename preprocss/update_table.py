import os, re, json, glob

json_files = glob.glob("*.jsonl")


for json_file in json_files:
    with open(json_file, "r", encoding='utf-8') as jfile:
        data = json.load(jfile) 
    basename = os.path.splitext(os.path.basename(json_file))[0]
    text_file = basename + "_tables_texts.txt"
       
    if os.path.exists(text_file):
        with open(text_file, "r", encoding='utf-8') as tfile:
            texts = tfile.read()

        tables = re.split(r"Table \d+:", texts.strip())

        for i, table in enumerate(tables):
            if table.strip():  # Ensure there is content in the table
                pattern = f"Table-{i}-to-be-replaced" 
                print(f"table:{i} \n {table}")
                data['text'] = re.sub(pattern, table, data['text'])

        filename = basename + "_updated.jsonl"

        if not os.path.exists(filename):
            with open(filename, 'w', encoding='utf-8') as jfile:
                json.dump(data, jfile, ensure_ascii=False, indent=4)        
