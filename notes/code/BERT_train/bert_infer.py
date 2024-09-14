import torch
from transformers import BertTokenizer, BertForQuestionAnswering
from torch.utils.data import DataLoader, TensorDataset
import json
from tqdm import tqdm
import time


model = BertForQuestionAnswering.from_pretrained('bert-large-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')


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


time1 = time.time()
questions, contexts, answers = prepare_data(squad_val_data)


encoding = tokenizer(contexts, questions, truncation=True, padding=True, return_tensors='pt')


dataset = TensorDataset(encoding['input_ids'], encoding['attention_mask'], encoding['token_type_ids'])
dataloader = DataLoader(dataset, batch_size=16)
print("Data encoding and loading time: ",time.time()-time1)
time2 = time.time()
model.eval()
model.to('cuda')
with torch.no_grad():
    for batch in tqdm(dataloader):
        input_ids, attention_mask, token_type_ids = [item.to('cuda') for item in batch]
        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        start_logits, end_logits = outputs.start_logits.to('cuda'), outputs.end_logits.to('cuda')
        # Convert logit scores to text (if required)
        for i in range(len(start_logits)):
            start = torch.argmax(start_logits[i])
            end = torch.argmax(end_logits[i])
            answer_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[i][start:end+1]))
            print(answer_text)
print("Total inference time: ", time.time()-time2)