import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-q","--question",type=str)


## v2 models
#model_path = 'openlm-research/open_llama_7b_v2'
model_path = 'jphme/Llama-2-13b-chat-german'

## v1 models
# model_path = 'openlm-research/open_llama_3b'
# model_path = 'openlm-research/open_llama_7b'
# model_path = 'openlm-research/open_llama_13b'

tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map='auto',
)
args = parser.parse_args()

if args.question is not None:
	question = args.question
else:
	question = "Bist du dumm oder was?"

prompt = "Frage: "+question
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

generation_output = model.generate(
    input_ids=input_ids, max_new_tokens=40
)
print(tokenizer.decode(generation_output[0]))

