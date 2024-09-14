import json
import os
from transformers import BertTokenizer
import torch

# Check if the preprocessed data file exists
preprocessed_file = "preprocessed_infer.json"
if not os.path.exists(preprocessed_file):
    # Preprocess and save the data
    with open("dev-v2.0.json", 'r') as f:
        squad_val_data = json.load(f)['data']

    def prepare_data(data):
        questions, contexts, answers = [], [], []
        for article in data:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    question = qa['question']
                    answer = qa['answers'][0]['text'] if qa['answers'] else ''
                    questions.append(question)
                    contexts.append(context)
                    answers.append(answer)
        return questions, contexts, answers
    
    questions, contexts, answers = prepare_data(squad_val_data)

    # Save the preprocessed data
    with open(preprocessed_file, 'w') as f:
        json.dump({"questions": questions, "contexts": contexts, "answers": answers}, f)
else:
    # Load the preprocessed data
    with open(preprocessed_file, 'r') as f:
        data = json.load(f)
        questions, contexts, answers = data["questions"], data["contexts"], data["answers"]

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
encoding = tokenizer(contexts, questions, truncation=True, padding=True, return_tensors='pt')
torch.save(encoding, "tokenized_infer.pt")