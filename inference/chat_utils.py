# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Literal, Optional, Tuple, TypedDict, Union
import json
import torch

Role = Literal["user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
def format_tokens(dialogs, tokenizer):
    prompt_tokens = []
    for dialog in dialogs:
        if dialog[0]["role"] == "system":
            dialog = [
            {
                "role": dialog[1]["role"],
                "content": B_SYS
                + dialog[0]["content"]
                + E_SYS
                + dialog[1]["content"],
            }
        ] + dialog[2:]
        assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
            [msg["role"] == "assistant" for msg in dialog[1::2]]
        ), (
            "model only supports 'system','user' and 'assistant' roles, "
            "starting with user and alternating (u/a/u/a/u...)"
        )
        """
        Please verify that your tokenizer support adding "[INST]", "[/INST]" to your inputs.
        Here, we are adding it manually.
        """
        dialog_tokens: List[int] = sum(
            [
                tokenizer.encode(
                    f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                )
                for prompt, answer in zip(dialog[::2], dialog[1::2])
            ],
            [],
        )
        assert (
            dialog[-1]["role"] == "user"
        ), f"Last message must be from user, got {dialog[-1]['role']}"
        dialog_tokens += tokenizer.encode(
            f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
        )
        prompt_tokens.append(dialog_tokens)
    return prompt_tokens
        

def read_dialogs_from_file(file_path):
    with open(file_path, 'r') as file:
        dialogs = json.load(file)
    return dialogs


def create_batches(input_ids_list, batch_size):
    # Initialize lists to store the batches of input_ids and attention_masks
    input_ids_batches = []
    attention_mask_batches = []
    
    # Create batches
    for i in range(0, len(input_ids_list), batch_size):
        # Get the current batch of input_ids
        input_ids_batch = input_ids_list[i:i+batch_size]
        max_length = max(len(input_ids) for input_ids in input_ids_batch)
        
        # Pad the input_ids and create attention masks
        padded_input_ids_batch = []
        attention_mask_batch = []
        for input_ids in input_ids_batch:
            # Pad input_ids to max_length
            padded_input_ids =  [0] * (max_length - len(input_ids)) + input_ids
            # Create attention_mask
            attention_mask =  [0] * (max_length - len(input_ids)) + [1] * len(input_ids)
            # Append to the batch lists
            padded_input_ids_batch.append(padded_input_ids)
            attention_mask_batch.append(attention_mask)
        
        # Convert to tensors and append to the output lists
        input_ids_batches.append(torch.tensor(padded_input_ids_batch))
        attention_mask_batches.append(torch.tensor(attention_mask_batch))
    
    return input_ids_batches, attention_mask_batches