import torch
from transformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer
model_name = 'meta-llama/Llama-2-13b-chat-hf'
tokenizer = LlamaTokenizer.from_pretrained(model_name)
config = LlamaConfig.from_pretrained(model_name, output_attentions=True, return_dict=True)
model = LlamaForCausalLM.from_pretrained(model_name,  device_map="auto", load_in_8bit=True, low_cpu_mem_usage=True, config=config)
# tokenizer.add_special_tokens({"pad_token": "<PAD>"})

# txt='You are an intelligent, honest, and harmless assistant. Your direct, concise responses contain only the minimum words needed to convey the answer. Provide your best guess and the probability that it is correct (0.0 to 1.0) for the following question. Give ONLY the guess and probability, no other words or explanation.\nAnswer format:\n\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>\nProbability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>\n\nThe question is: What general name is given to a rotating star which emits a regular beat of radiation?'
# txt='You are an intelligent, honest, and harmless assistant. Your direct, concise responses contain only the minimum words needed to convey the answer. Provide your best guess for the following question. Give ONLY the guess, no other words or explanation.\nAnswer format:\n\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>\n\nThe question is: What general name is given to a rotating star which emits a regular beat of radiation?'
txt='Answer format:\n\nGuess: <a short answer>\n\nWhich 19th century British novelist, who worked for the Post Office, was reputedly instrumental in the introduction of the pillar box in 1852?'
txt='Answer format:\n\nGuess: <a short answer>\n\nWhich 19th century British, who worked for the Post Office, was reputedly instrumental in the introduction of the pillar box in?'
txt='Answer format:\n\nGuess: <a short answer>\n\nWhich 19th century British, was reputedly instrumental in the introduction of the pillar box?'
# txt='Answer format:\n\nGuess: <a short answer>\n\n Kirkland <unk> is the house brand of what retail giant?'
inputs = tokenizer([txt],return_tensors='pt').to('cuda:0')
input_length = inputs.input_ids.shape[1]
outputs = model.generate(**inputs, output_attentions=True, return_dict_in_generate=True, max_new_tokens=100, do_sample=True, top_k=50, temperature=0.7)
# transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
# print(f' attention size: {len(outputs.attentions)}') 
# print(f' attention size: {len(outputs.attentions[-1])}') 
# print(f' attention size: {len(outputs.attentions[-1][-1])}') 
# print(f' attention size: {len(outputs.attentions[-1][-1][0])}') 
# print(f' attention size: {len(outputs.attentions[-1][-1][0][-1])}') 
# print(f' attention size: {len(outputs.attentions[-1][-1][0][-1][-1])}') 
# print(f' attention size: {len(outputs.attentions[-1][-1][0][-1][-1][-1])}')
# print(f' attention size: {type(outputs.attentions)}') 
# print(f' attention size: {type(outputs.attentions[-1])}') 
# print(f' attention size: {type(outputs.attentions[-1][-1])}') 
# print(f' attention size: {type(outputs.attentions[-1][-1][0])}') 
# print(f' attention size: {type(outputs.attentions[-1][-1][0][-1])}') 
# print(f' attention size: {type(outputs.attentions[-1][-1][0][-1][-1])}') 
# print(f' attention size: {type(outputs.attentions[-1][-1][0][-1][-1][-1])}')
# print(f' att vector size: {len(outputs.attentions)}') 
# print(f' att vector size: {len(outputs.attentions[-1])}') 
# print(f' att vector size: {len(outputs.attentions[-1][-1])}') 
# print(f' att vector size: {len(outputs.attentions[-1][-1][1])}') 
# print(f' att vector size: {len(outputs.attentions[-1][-1][1][-1])}') 
# print(f' att vector size: {len(outputs.attentions[-1][-1][1][-1][-1])}') 
# print(f' att vector size: {type(outputs.attentions)}') 
# print(f' att vector size: {type(outputs.attentions[-1])}') 
# print(f' att vector size: {type(outputs.attentions[-1][-1])}') 
# print(f' att vector size: {type(outputs.attentions[-1][-1][1])}') 
# print(f' att vector size: {type(outputs.attentions[-1][-1][1][-1])}') 
# print(f' att vector size: {type(outputs.attentions[-1][-1][1][-1][-1])}') 



input_tokens = tokenizer.tokenize(txt)
print(input_tokens)
input_seq_len = len(input_tokens)

generated_tokens = outputs.sequences[:, input_length:]
response = tokenizer.decode(generated_tokens[0])
answer_len = len(tokenizer.tokenize('Guess: '.join(response.split('Guess: ')[1:])))
answer_token_index = -answer_len

prompt_end_index = len(tokenizer.tokenize(txt.split('<a short answer>\n\n')[0]+'<a short answer>\n\n'))


answer_token_outputs = outputs.attentions[answer_token_index]
attention_weights = torch.cat([x[0] for x in answer_token_outputs], dim=0)
attention_weights = torch.squeeze(attention_weights) # 40 x 40 x (seq_len + 1)
attention_weights = attention_weights[:, :, prompt_end_index + 1:]
print(f'answer token attention weight size: {attention_weights.size()}')
attention_vectors = torch.stack([x[1] for x in answer_token_outputs], dim=0)
attention_vectors = torch.squeeze(attention_vectors) # 40 x (seq_len + 1) x 5120
attention_vectors = attention_vectors[:, prompt_end_index + 1:, :]
print(f'answer token attention vector size: {attention_vectors.size()}')


total_len = attention_weights.size()[-1]
max_head_att = torch.max(attention_weights, dim=1).values
max_layer_max_head_att = torch.max(max_head_att, dim=0).values
norm_head_att = torch.norm(attention_weights, dim=1)
max_layer_norm_head_att = torch.max(norm_head_att, dim=0).values
att_vector_norm = torch.norm(attention_vectors, dim=2)
max_layer_vector_norm = torch.max(att_vector_norm, dim=0).values


token_labels = []
print(f"input length: {input_seq_len}")
print(f"output length: {len(generated_tokens[0])}")
print(f"answer index: {answer_token_index}")
print(f"attention size at answer index: {outputs.attentions[answer_token_index][0][0].size()}")
print(f"total length: {total_len}")
print(f"prompt end index: {prompt_end_index}")
for i in range(total_len):
    if prompt_end_index + i == input_seq_len:
        break
    tok = input_tokens[prompt_end_index + i]
    print(f"{tok:8s} | {max_layer_max_head_att[i]:.3f} | {max_layer_norm_head_att[i]:.3f} | {max_layer_vector_norm[i]:.3f}")
    token_labels.append(tok)


print(tokenizer.decode(generated_tokens[0]))
print(f"answer at {answer_token_index}: {tokenizer.decode(generated_tokens[0][answer_token_index-1])}|{tokenizer.decode(generated_tokens[0][answer_token_index])}|{tokenizer.decode(generated_tokens[0][answer_token_index+1])}")
for i, tok in enumerate(generated_tokens[0]):
    j = i + input_seq_len - prompt_end_index
    if j == total_len:
        break
    print(f"{tokenizer.decode(tok):8s} | {max_layer_max_head_att[j]:.3f} | {max_layer_norm_head_att[j]:.3f}")
    token_labels.append(tokenizer.decode(tok))


# For drawing figures:
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib as mpl

fig, ax = plt.subplots(figsize=(15, 15))
im = ax.imshow(max_head_att.detach().cpu().numpy())  # 40 x seq_len
cbar = ax.figure.colorbar(im, ax=ax)

ax.set_xticks(np.arange(len(token_labels)), labels=token_labels, rotation=90, fontsize=22)
# ax.set_yticklabels([str(x) for x in [0,5,10,15,20,25,30,35]], fontsize=22)
ax.set_title('max att over heads',fontsize=22)
fig.tight_layout()
plt.show()
plt.savefig('max_head.png')


fig, ax = plt.subplots(figsize=(15, 15))
im = ax.imshow(norm_head_att.detach().cpu().numpy())  # 40 x seq_len
cbar = ax.figure.colorbar(im, ax=ax)

ax.set_xticks(np.arange(len(token_labels)), labels=token_labels, rotation=90, fontsize=22)
# ax.set_yticklabels([str(x) for x in [0,5,10,15,20,25,30,35]], fontsize=22)
ax.set_title('norm. att over heads',fontsize=22)
fig.tight_layout()
plt.show()
plt.savefig('norm_head.png')


fig, ax = plt.subplots(figsize=(15, 15))
im = ax.imshow(att_vector_norm.detach().cpu().numpy())  # 40 x seq_len
cbar = ax.figure.colorbar(im, ax=ax)

ax.set_xticks(np.arange(len(token_labels)), labels=token_labels, rotation=90, fontsize=22)
# ax.set_yticklabels([str(x) for x in [0,5,10,15,20,25,30,35]], fontsize=22)
ax.set_title('attention contribution norm',fontsize=22)
fig.tight_layout()
plt.show()
plt.savefig('att_vector_norm.png')




