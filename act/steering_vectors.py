from einops import rearrange, repeat, einsum
import gc
import math
import numpy as np
import torch
from tqdm import tqdm

import utils

def find_steering_vecs(base_toks, target_toks, model, layer, batch_size = 16, pos=-1, get_raw_diffs = False): 
    '''
    We want to find the steering vector from base_toks to target_toks (we do target_toks - base_toks)
    Inputs: 
        :param model: the model to use
        :param base_toks: the base tokens [len, seq_len]
        :param target_toks: the target tokens [len, seq_len]
    Output: 
        :return steering_vecs: the steering vectors [hidden_size] or [seq_len, hidden_size]
    '''
    assert base_toks.shape == target_toks.shape, "Base and target tokens must have the same shape"
    device = model.device
    num_its = len(range(0, base_toks.shape[0], batch_size))
    steering_vecs = {}
    raw_diffs = {}

    activations = {}
    def get_activation(name):
        def hook(model, input):
            activations[name] = input[0]
        return hook

    for i in tqdm(range(0, base_toks.shape[0], batch_size)): 
        # add the hook
        h = model.model.layers[layer].register_forward_pre_hook(get_activation(layer))
        # get the activations
        model(base_toks[i:i+batch_size].to(device))
        base_out = activations[layer]
        model(target_toks[i:i+batch_size].to(device))
        target_out = activations[layer]
        # average over the batch_size, take last token 
        if i == 0: 
            if pos == None:
                if get_raw_diffs: 
                    raw_diffs = target_out.detach().cpu() - base_out.detach().cpu()
                steering_vecs = torch.mean(target_out.detach().cpu() - base_out.detach().cpu(), dim=0)/num_its # [seq_len, hidden_size]
            else: 
                if get_raw_diffs:
                    raw_diffs = target_out[:,pos,:].detach().cpu() - base_out[:,pos,:].detach().cpu()
                steering_vecs = torch.mean(target_out[:,pos,:].detach().cpu() - base_out[:,pos,:].detach().cpu(), dim=0)/num_its # [hidden_size]
        else: 
            if pos == None:
                if get_raw_diffs:
                    raw_diffs = torch.cat([raw_diffs, target_out.detach().cpu() - base_out.detach().cpu()], dim=0)
                steering_vecs += torch.mean(target_out.detach().cpu() - base_out.detach().cpu(), dim=0)/num_its
            else: 
                if get_raw_diffs:
                    raw_diffs = torch.cat([raw_diffs, target_out[:,pos,:].detach().cpu() - base_out[:,pos,:].detach().cpu()], dim=0)
                steering_vecs += torch.mean(target_out[:,pos,:].detach().cpu() - base_out[:,pos,:].detach().cpu(), dim=0)/num_its
        # remove the hook
        h.remove()
    return steering_vecs, raw_diffs 

def optimise_steering_vecs(model, toks_A, toks_B, hidden_dim, layer, type='full', batch_size=6, num_epochs=10): 
    '''
    This method runs an optimisation over the steering vectors to steer towards data_A and away from data_B
    It minimises a cross-entropy loss towards data_A and maximises a cross-entropy loss away from data_B
    Inputs:
        :param model: the model to use
        :param toks_A: the data to steer towards
        :param toks_B: the data to steer away from
        :param hidden_dim: the hidden dimension of the model
        :param layer: the layer to get the representation from
        :param type: the type of steering vector to use: 'full', 'constant', 'reft'
        :param batch_size: the batch size to use
        :param num_epochs: the number of epochs to run
    Output:
        :return steering_vec: the optimised steering vector
    Comments: 
        - The constant and reft methods aren't fully optimised yet! Use with care :) 
    '''
    device = model.device
    # freeze model params 
    for param in model.parameters():
        param.requires_grad = False
    # initialise trainable parameter: the steering vector to add to the layer of choice 
    if type=='full':
        steering_vec = torch.randn(hidden_dim, requires_grad=True, dtype=torch.bfloat16)
    elif type=='constant':
        steering_vec = torch.randn(1, requires_grad=True, dtype=torch.bfloat16)
    elif type=='reft':
        r_dim = 4 # representation dimension
        stdv = 1. / math.sqrt(hidden_dim)
        R_opt = torch.nn.Parameter(torch.empty(r_dim, hidden_dim), requires_grad=True)
        torch.nn.init.orthogonal_(R_opt)
        W_opt = torch.nn.Parameter(torch.empty(r_dim, hidden_dim), requires_grad=True)
        torch.nn.init.uniform_(W_opt, -stdv, stdv)
        b_opt = torch.nn.Parameter(torch.empty(r_dim), requires_grad=True)
        torch.nn.init.uniform_(b_opt, -stdv, stdv)
    # add steering vector to layer with a hook
    def sv_hook(module, input):
        if type=='full':
            modified_input = (input[0] + steering_vec.to(device),)
        elif type=='constant':
            modified_input = (input[0] * (1+steering_vec.to(device)),) 
        elif type=='reft':
            # from: https://github.com/stanfordnlp/pyreft/blob/main/pyreft/interventions.py
            # TO DO: improve initialisation
            # R^T (Wh + b − Rh), R
            adjustment = torch.einsum('rh,blh->blr', W_opt.to(torch.bfloat16).to(device), input[0]) + b_opt.to(torch.bfloat16).to(device) - torch.einsum('rh,blh->blr', R_opt.to(torch.bfloat16).to(device), input[0])
            modified_input = (input[0] + torch.einsum('rh,blr->blh', R_opt.to(torch.bfloat16).to(device), adjustment),)
        return modified_input
    h_o = model.model.layers[layer].register_forward_pre_hook(sv_hook)
    if type=='reft':
        optimiser = torch.optim.Adam([R_opt, W_opt, b_opt], lr=0.01)
    else:
        optimiser = torch.optim.Adam([steering_vec], lr=0.01)
    for epoch in range(num_epochs): 
        # iterate over dataset of chosen style 
        for i in range(0, toks_A.shape[0], batch_size): 
            optimiser.zero_grad()
            # pass train data through model 
            out_A = model(toks_A[i:i+batch_size,:-1])
            # calculate loss 
            loss_A = torch.nn.functional.cross_entropy(out_A.logits.transpose(1,2), toks_A[i:i+batch_size,1:])
            # steer towards A, steer away from B
            threshold = 3
            loss = loss_A
            if toks_B is not None:
                out_B = model(toks_B[i:i+batch_size,:-1])
                loss_B = torch.nn.functional.cross_entropy(out_B.logits.transpose(1,2), toks_B[i:i+batch_size,1:])
                loss += torch.max(torch.tensor(0.0).to(device), threshold - loss_B)
            print(i, loss.item())
            # backpropagate
            loss.backward()
            # update steering vector
            optimiser.step()
            del out_A, loss_A
            if toks_B is not None:
                del out_B, loss_B
            torch.cuda.empty_cache()
            gc.collect()
    h_o.remove()
    if type=='reft':
        return (R_opt, W_opt, b_opt)
    else: 
        return steering_vec

def do_steering(model, test_toks, steering_vec, steer_type = 'full', scale = 1, layer = None, proj=True, all_toks = False, batch_size=16): 
    '''
    Given a steering)vec, we steer the model output
    Input: 
        :param model: the model to use
        :param test_toks: the test tokens [len, seq_len]
        :param steering_vec: 
            if type=='full': the steering vector [hidden_size] or [seq_len, hidden_size]
            if type=='constant': the steering scalar [1]
            if type=='reft': a tuple of steering vectors [W_opt, R_opt, b_opt]
        :param scale: the scale to use
        :param layer: the (list of) layer(s) to modify; if None: we modify all layers.  
        :param proj: whether to project the steering vector
    Output:
        :return output: the steered model output [len, generated_seq_len]
    '''
    
    # A bunch of checks 
    if steer_type == 'reft':
        assert len(steering_vec) == 3, "For reft, we need to pass a tuple of steering vectors [W_opt, R_opt, b_opt]"
    if steer_type == 'constant' or steer_type == 'reft':
        assert all_toks == False, "For reft or constant, we do not intervene on all tokens"
        assert proj == False, "For reft or constant, we do not project the steering vector"
    if all_toks == True: 
        assert len(steering_vec.shape) == 2, "If you want to intervene on all tokens, you need to pass something of shape [seq_len, hidden_size]"
        assert proj == False, "We do not project if steering vector is of shape [seq_len, hidden_size]" 
    
    # Define a hook to modify the input into the layer
    if steering_vec is not None: 
        if steer_type == 'full': 
            def modify_activation():
                def hook(model, input): 
                    if proj:
                        sv = steering_vec / steering_vec.norm()
                        sv = einsum(input[0], sv.view(-1,1), 'b l h, h s -> b l s') * sv # shape [batch_size, seq_len, hidden_size]
                    else: 
                        sv = steering_vec
                    if not all_toks: 
                        input[0][:,:,:] = input[0][:,:,:] - scale * sv
                    else:
                        if input[0].shape[1] > 1: # we only intervene on the first pass
                            len_refusal = min(steering_vec.shape[0], input[0].shape[1])
                            input[0][:,:len_refusal,:] = input[0][:,:len_refusal,:] - scale * sv.unsqueeze(0).repeat(input[0].shape[0],1,1)[:,:len_refusal,:]  
                        else: 
                            input[0][:,:,:] = input[0][:,:,:]
                return hook
        elif steer_type == 'constant':
            def modify_activation():
                def hook(model, input): 
                    input[0][:,:,:] = steering_vec * input[0][:,:,:]
                return hook
        elif steer_type == 'reft':
            W_opt, R_opt, b_opt = steering_vec[0], steering_vec[1], steering_vec[2]
            def modify_activation():
                def hook(model, input): 
                    adjustment = torch.einsum('rh,blh->blr', W_opt.to(torch.bfloat16), input[0]) + b_opt.to(torch.bfloat16) - torch.einsum('rh,blh->blr', R_opt.to(torch.bfloat16), input[0])
                    input[0][:,:,:] = input[0][:,:,:] + torch.einsum('rh,blr->blh', R_opt.to(torch.bfloat16), adjustment)
                return hook
        handles = [] 
        for i in range(len(model.model.layers)):
            if layer is None: # append to each layer
                handles.append(model.model.layers[i].register_forward_pre_hook(modify_activation()))
            elif layer is not None and i == layer:
                handles.append(model.model.layers[i].register_forward_pre_hook(modify_activation()))
            elif layer is not None and type(layer) == list and i in layer:
                handles.append(model.model.layers[i].register_forward_pre_hook(modify_activation()))
    # pass through the model
    outs_all = []
    topk_all = []
    for i in range(0, test_toks.shape[0], batch_size):
        # during generation input shape is [1, seq_len, hidden_size] in 1st pass and after it [1, 1, hidden_size]
        # outs, top_probas = utils.custom_generate(model, test_toks[i:i+batch_size], max_new_tokens=150, temperature=0.5, k=5)
        outs = model.generate(test_toks[i:i+batch_size], max_new_tokens=400)
        outs_all.append(outs)
        # topk_all.append(top_probas)
    outs_all = torch.cat(outs_all, dim=0)
    # remove all hooks
    if steering_vec is not None: 
        for handle in handles: 
            handle.remove()
    return outs_all, topk_all
