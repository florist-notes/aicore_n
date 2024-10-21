from transformers import AutoTokenizer, AutoModelForCausalLM

# Specify the model you want to download
model_name = "meta-llama/Llama-3B"  # Replace with the actual LLaMA 3 model ID

# Download and cache the model locally
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Save the model to a local folder (optional)
model.save_pretrained('./local_llama3_model')
tokenizer.save_pretrained('./local_llama3_model')
