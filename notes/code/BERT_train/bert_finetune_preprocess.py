import json
import os
from transformers import BertTokenizerFast
import torch

preprocessed_file = "preprocessed_finetune.json"
tokenized_file = "tokenized_finetune.pt"

if not os.path.exists(preprocessed_file):
    # Load the dataset
    with open("train-v2.0.json", 'r') as f:
        squad_val_data = json.load(f)['data']

    # Prepare the data
    questions, contexts, answers, start_positions = [], [], [], []
    for article in squad_val_data:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                if qa['answers']:
                    answer = qa['answers'][0]
                    answers.append(answer['text'])
                    start_positions.append(answer['answer_start'])
                else:
                    answers.append("")  # No answer
                    start_positions.append(0)  # Placeholder position
                questions.append(question)
                contexts.append(context)

    # Tokenize and align the start and end positions
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    start_tokens = []
    end_tokens = []

    for i in range(len(questions)):
        # Tokenize context and question
        encodings = tokenizer(contexts[i], questions[i], truncation=True, padding='max_length', return_offsets_mapping=True, return_tensors='pt')
        input_ids = encodings['input_ids'].squeeze()
        attention_mask = encodings['attention_mask'].squeeze()
        token_type_ids = encodings['token_type_ids'].squeeze()

        # Get start and end token positions
        answer = answers[i]
        start_char = start_positions[i]
        end_char = start_char + len(answer) - 1

        # Find the start and end of the answer in the tokens
        sequence_ids = encodings.sequence_ids()
        offsets = encodings['offset_mapping'].squeeze().tolist()
        start_token = next((i for i, (offset, seq) in enumerate(zip(offsets, sequence_ids)) if seq == 1 and offset[0] <= start_char < offset[1]), None)
        end_token = next((i for i, (offset, seq) in enumerate(zip(offsets, sequence_ids)) if seq == 1 and offset[0] < end_char <= offset[1]), None)

        # If the answer cannot be found in the tokens, mark the cls index
        if start_token is None:
            start_token = 0
        if end_token is None:
            end_token = start_token

        start_tokens.append(start_token)
        end_tokens.append(end_token)

        # Save the data for this example
        if i == 0:
            all_input_ids = input_ids.unsqueeze(0)
            all_attention_masks = attention_mask.unsqueeze(0)
            all_token_type_ids = token_type_ids.unsqueeze(0)
        else:
            all_input_ids = torch.cat((all_input_ids, input_ids.unsqueeze(0)), dim=0)
            all_attention_masks = torch.cat((all_attention_masks, attention_mask.unsqueeze(0)), dim=0)
            all_token_type_ids = torch.cat((all_token_type_ids, token_type_ids.unsqueeze(0)), dim=0)

    # Save the preprocessed data
    with open(preprocessed_file, 'w') as f:
        json.dump({
            "questions": questions,
            "contexts": contexts,
            "answers": answers,
            "start_positions": start_positions
        }, f)

    # Convert start and end tokens to tensors
    all_start_positions = torch.tensor(start_tokens, dtype=torch.long)
    all_end_positions = torch.tensor(end_tokens, dtype=torch.long)

    # Save the tokenized data and positions
    torch.save({
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "token_type_ids": all_token_type_ids,
        "start_positions": all_start_positions,
        "end_positions": all_end_positions
    }, tokenized_file)

else:
    # Load the preprocessed data
    tokenized_data = torch.load(tokenized_file)
