import json, re
# handle multiline answers 

def cleanup(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            line = line.replace("Question:", "").replace("Question", "").replace("Answer:", "").replace("Answer", "").strip()
            line = re.sub(r'^\d{1,2}\. A:', "A:", line)
            line = re.sub(r'^\d{1,2}\. Q:', "Q:", line) 
            line = re.sub(r'^A\d{1,2}:', "A:", line)
            line = re.sub(r'^Q\d{1,2}:', "Q:", line) 
            line = re.sub(r'^Sure', "TO-BE-REMOVE", line) 

            if "email exchange" in line or "a list of instruction" in line:
                continue  

            if "TO-BE-REMOVE" in line or "instruction (Q:)" in line:
                continue  

            if "I hope these pairs" in line or "Diversity" in line: 
                continue 

            if "repeat the verb" in line or "diversity" in line or "diverse" in line: 
                continue 

            if " A:" in line:
                line = line.replace("A:", "\nA:")
            if " Q:" in line:
                line = line.replace("Q:", "\nQ:")

            outfile.write(line + '\n')


def extract_qa_to_json_lines(input_file, output_file):
    current_question = None
    current_answer = None

    json_lines = []

    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()

            if line.startswith("Q:"):
                if current_question is not None and len(current_answer[3:-2]) >0:
                    json_line = {
                        "prompt": f"Human: {current_question} Assistant:",
                        "chosen": rf'{current_answer[3:-2]}'
                    }
                    json_lines.append(json_line)

                current_question = line[2:].strip()
                current_answer = ""
            elif current_question is not None:
                current_answer += line + '\n'

    if current_question is not None:
        json_line = {
            "prompt": f"Human: {current_question} Assistant:",
            "chosen": current_answer[3:]
        }
        json_lines.append(json_line)

    with open(output_file, 'w', encoding='utf-8') as output_file:
        for json_line in json_lines:
            output_file.write(json.dumps(json_line) + '\n')




def extract_qa_to_json_lines_v1(input_file, output_file):
    qa_pairs = []

    with open(input_file, 'r') as file:
        lines = file.readlines()

    current_q = ""
    for line in lines:
        line = line.strip()
        if line.startswith("Q:"):
            if current_q:
                if "knowledge base" not in current_q and "knowledge base" not in current_a:
                    qa_pairs.append({
                        "prompt": f"Human: {current_q[3:]} Assistant:",
                        "chosen": current_a[3:]
                    })
            current_q = line
            current_a = ""
        elif line.startswith("A:"):
            current_a = line
    
    if current_q:
        if "knowledge base" not in current_q and "knowledge base" not in current_a:
            qa_pairs.append({
                "prompt": f"Human: {current_q[3:]} Assistant:",
                "chosen": current_a[3:]
            })

    with open(output_file, 'w', encoding='utf-8') as json_file:
        for qa_pair in qa_pairs:
            json_line = json.dumps(qa_pair)
            json_file.write(json_line + '\n')

input_file = "jira.txt"  
clean_file = "output.clean"  
output_file = "output.jsonl"  

cleanup(input_file, clean_file)
extract_qa_to_json_lines(clean_file, output_file)

