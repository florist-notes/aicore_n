from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")

# Example text
text = "Replace me by any text you'd like."

# Tokenize and encode the input text
encoded_input = tokenizer(text, return_tensors='pt')

# Forward pass through the model
with torch.no_grad():
    output = model(**encoded_input)

# Function to calculate FLOPs for self-attention mechanism
def calculate_flops_self_attention(attention_heads, sequence_length):
    # Assuming attention mechanism involves matrix multiplications
    return 2 * attention_heads * sequence_length ** 2

# Calculate and print FLOPs
flops = 0
print("FLOPs calculation:")
for name, module in model.named_children():
    if 'self' in name and 'attn' in name:  # Assuming attention layers
        attention_heads = module.num_attention_heads
        sequence_length = encoded_input['input_ids'].shape[1]
        flops_layer = calculate_flops_self_attention(attention_heads, sequence_length)
        flops += flops_layer
        print(f"{name}: {flops_layer}")

print(f"\nTotal FLOPs: {flops}")
