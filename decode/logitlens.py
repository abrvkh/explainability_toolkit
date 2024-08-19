import numpy as np
import torch 
from tqdm import tqdm

def logitlens_decode_token(model, device, tokenizer, input): 
    '''
    Decodes intermediate tokens with LogitLens 
    Args: 
    - model: model with output_hidden_states and output_attentions
    - device: device on which you are working 
    - tokenizer: tokenizer for the model
    - input: input text to decode: str; e.g. this would be the output from a generation inside labs
    Output: 
    - a dictionary with {'decoded_tokens': [num_layers, seq_len], 'decoded_logits': [num_layers, seq_len]}
    '''
    inputs = tokenizer(input, return_tensors="pt").to(device)
    text_tokens = [tokenizer.decode(id) for id in inputs['input_ids'][0]]
    
    # apply decoder lens
    classifier_head = model.lm_head # Linear(in_features=3072, out_features=32064, bias=False)

    hidden_states = model(**inputs, output_hidden_states = True).hidden_states
    decoded_intermediate_token = {}
    decoded_intermediate_logit = {}
    with torch.no_grad():
        for layer_id in range(len(hidden_states)): 
            hidden_state = hidden_states[layer_id]
            decoded_value = classifier_head(hidden_state) # [batch, seq_len, vocab_size]
            # get probabilities
            decoded_values = torch.nn.functional.softmax(decoded_value, dim=-1)
            # take max element
            argmax = torch.argmax(decoded_values, dim=-1)[0] # select first element in batch
            # decode all tokens
            decoded_token = [tokenizer.decode(el) for el in argmax]
            decoded_logit = [decoded_values[0, it, argmax[it]].item() for it in range(len(argmax))] # list of layers, per layer the sequence_length
            decoded_intermediate_token[layer_id] = decoded_token
            decoded_intermediate_logit[layer_id] = decoded_logit

    return {'text_tokens':text_tokens, 'decoded_tokens': decoded_intermediate_token, 'decoded_logits': decoded_intermediate_logit}
