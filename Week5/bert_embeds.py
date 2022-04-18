from transformers import BertTokenizer , BertModel
import os
import torch
import json
import random
import numpy as np


def bert_text_preparation(text, tokenizer):
    """Preparing the input for BERT
    
    Takes a string argument and performs
    pre-processing like adding special tokens,
    tokenization, tokens to ids, and tokens to
    segment ids. All tokens are mapped to seg-
    ment id = 1.
    
    Args:
        text (str): Text to be converted
        tokenizer (obj): Tokenizer object
            to convert text into BERT-re-
            adable tokens and ids
        
    Returns:
        list: List of BERT-readable tokens
        obj: Torch tensor with token ids
        obj: Torch tensor segment ids
    
    
    """
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1]*len(indexed_tokens)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    return tokenized_text, tokens_tensor, segments_tensors
    
    
def get_bert_embeddings(tokens_tensor, segments_tensors, model):
    """Get embeddings from an embedding model
    
    Args:
        tokens_tensor (obj): Torch tensor size [n_tokens]
            with token ids for each token in text
        segments_tensors (obj): Torch tensor size [n_tokens]
            with segment ids for each token in text
        model (obj): Embedding model to generate embeddings
            from token and segment ids
    
    Returns:
        list: List of list of floats of size
            [n_tokens, n_embedding_dimensions]
            containing embeddings for each token
    
    """
    
    # Gradient calculation id disabled
    # Model is in inference mode
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        # Removing the first hidden state
        # The first state is the input state
        hidden_states = outputs[2][1:]

    # Getting embeddings from the final BERT layer
    token_embeddings = hidden_states[-1]
    # Collapsing the tensor into 1-dimension
    token_embeddings = torch.squeeze(token_embeddings, dim=0)
    # Converting torchtensors to lists
    list_token_embeddings = [token_embed.tolist() for token_embed in token_embeddings]

    return list_token_embeddings


bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)

pth = "/export/home/mcv/datasets/Flickr30k/"
agg = "sum"

os.makedirs("bert_embeds", exist_ok=True)

for mode in ["train", "val", "test"]:
    print(mode)
    with open(os.path.join(pth, mode + ".json")) as f:
        labels = json.load(f)
        
    text_embeds = []     
    
    for i, lbl in enumerate(labels):
        label_tokens = []
        print(i)
        
        for sntc in lbl["sentences"]:
        
            tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(sntc["raw"], bert_tokenizer)
            #inputs = bert_tokenizer(sntc["raw"], return_tensors="pt")
            list_token_embeddings = get_bert_embeddings(tokens_tensor, segments_tensors, bert_model)
            #outputs = bert_model(**inputs)
            label_tokens.append(torch.tensor(list_token_embeddings))
            #label_tokens.append(outputs.last_hidden_state[0])
        
        max_length = (max([token.shape[0] for token in label_tokens]))

        padded_tokens = []
        for tkn in label_tokens:

            if tkn.shape[0] < max_length:
                padded_tokens.append(torch.cat((tkn, torch.zeros((max_length-tkn.shape[0], 768)))))
            else:
                padded_tokens.append(tkn)     
        
        if agg == "sum":
            text_embeds.append(torch.sum(torch.stack(padded_tokens), dim=0).flatten())
                
        elif agg == "avg":
            text_embeds.append(torch.avg(torch.stack(padded_tokens), dim=0).flatten())
            
        elif agg == "concat":
            text_embeds.append(torch.stack(padded_tokens).flatten())
            
        elif agg == "random":
            rnd_idx = random.randint(0, 4)
            text_embeds.append(torch.stack(padded_tokens)[rnd_idx].flatten())
        else:
            print("Aggregation not known!")
        #print(text_embeds[0].shape)
        
    max_length = max([embd.shape[0] for embd in text_embeds])
    
    padded_embeds = []
    for embd in text_embeds:
        if embd.shape[0] < max_length:
            padded_embeds.append(torch.cat((embd, torch.zeros((max_length-embd.shape[0])))))
        else:
            padded_embeds.append(embd)
    print(torch.stack(padded_embeds).to("cpu").detach().numpy().shape)    
    with open(os.path.join("bert_embeds", "_".join((mode, agg, "bert_embeds.npy"))), "wb") as f:
        np.save(f, torch.stack(padded_embeds).to("cpu").detach().numpy())
         