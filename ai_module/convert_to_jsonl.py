import json

def convert_json_to_jsonl(input_file, output_file):
    # Read the JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Write each conversation as a separate line in JSONL format
    with open(output_file, 'w', encoding='utf-8') as f:
        for conversation in data:
            # Convert the conversation to a single line
            json_line = json.dumps(conversation, ensure_ascii=False)
            f.write(json_line + '\n')

if __name__ == "__main__":
    input_file = "Fine-tuning-data/Cevap_Yapilandirma/Deneme2.json"
    output_file = "Fine-tuning-data/Cevap_Yapilandirma/Data_5.jsonl"
    convert_json_to_jsonl(input_file, output_file)
    print(f"Conversion completed. Output saved to {output_file}") 