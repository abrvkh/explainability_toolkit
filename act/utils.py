from einops import rearrange, repeat, einsum
import numpy as np
import torch
from tqdm import tqdm

def sample_softmax(preds, temperature=0.5, k = 5):
    """Helper function to sample an index from a probability array."""
    preds = torch.softmax(preds, dim=-1)
    preds = preds.cpu().numpy().astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.default_rng().multinomial(1, preds, 1)
    return np.argmax(probas).item()

def custom_generate(model, tokens, max_new_tokens=None, temperature=None, k=5): 
    '''
    A generation loop I use and where I can control how and where I add the steering
    '''
    model.eval() 
    # tokens = model.tokenizer.encode(prompt)
    num_orig_tokens = len(tokens)
    num_tokens_generated = 0
    top_probas = []
    try: 
        tokenizer_len = model.lm_head.out_features
    except:
        return
    with torch.no_grad(): 
        while True: 
            input_ids = tokens 
            #data = torch.LongTensor(input_ids).view(1, -1).to(model.device)
            output = model(input_ids).logits
            if isinstance(output, tuple):
                output = output[0] # [1, seq_len, vocab_size]; gives the logits
            next_token_pos = tokens.shape[1] - 1 # starts off at last token
            probas = output[0, next_token_pos]
            if temperature is None: # greedy sampling
                output_token = torch.argmax(probas).item() 
            else: # softmax sampling
                output_token = sample_softmax(probas, temperature)
            output_token = min(output_token, tokenizer_len - 1)
            tokens = torch.cat((tokens,torch.tensor(output_token).view(1,1).to(model.device)), dim=1)
            num_tokens_generated += 1
            # obtain also topk probas 
            probas = torch.softmax(probas, dim=-1)
            vals, idx = torch.topk(probas, k)
            top_probas.append((vals, idx)) # list of length num_tokens_generated with (vals, idx) tuples of shape [k], [k]
            if (max_new_tokens is not None and num_tokens_generated >= max_new_tokens):
                break
    return tokens, top_probas