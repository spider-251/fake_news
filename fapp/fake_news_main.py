import io
import os
import numpy as np
import torch
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader, Dataset
from ml_things import plot_dict,plot_confusion_matrix,fix_text
from sklearn.metrics import classification_report,accuracy_score
from transformers import set_seed,TrainingArguments,Trainer,GPT2Config, GPT2Tokenizer,AdamW,get_linear_schedule_with_warmup,GPT2ForSequenceClassification
from transformers import AutoModel

access_token = "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"

set_seed(111)
epochs = 15
batch_size =32
max_length =60
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name_or_path = 'gpt2'
labels_ids = {'legit':0,'fake':1}
n_labels = len(labels_ids)
os.makedirs('temp_text',exist_ok =True)


class FakeNewsData(DataLoader):

    def __init__(self,path,use_tokenizer):

        self.texts = []
        self.labels = []
        content = io.open(path,mode='r',encoding='utf-8').read()
        content = fix_text(content)
        self.texts.append(content)
        self.labels.append('fake')

        self.n_examples = 1

        return
    
    def __len__(self):
        return self.n_examples
            

    def __getitem__(self, item):
        return {'text':self.texts[item],
        'label':self.labels[item]}

    
    
    
    
class Gpt2ClassificationColllator(object):
    
    def __init__(self,use_tokenizer, labels_encoder, max_sequence_len = None):

        self.use_tokenizer = use_tokenizer
        self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
        self.labels_encoder = labels_encoder
        return

    def __call__(self,sequence):

        texts = [sequence['text'] for sequence in sequence]

        labels = [sequence['label'] for sequence in sequence]

        labels = [self.labels_encoder[label] for label in labels]

        inputs = self.use_tokenizer(text = texts, return_tensors = 'pt', padding =True, truncation =True, max_length =self.max_sequence_len)
        inputs.update({'labels':torch.tensor(labels)})

        return inputs
    
    
def validation(dataloader,device_,model):
#     global model

    predictions_labels = []
    true_labels = []
    total_loss = 0

    model.eval()

    for batch in dataloader:
        true_labels+= batch['labels'].numpy().flatten().tolist()

        batch = {k:v.type(torch.long).to(device_) for k,v in batch.items()}

        with torch.no_grad():

            outputs = model(**batch)

            loss,logits = outputs[:2]

            logits = logits.detach().cpu().numpy()

            total_loss+= loss.item()

            predict_content = logits.argmax(axis =-1).flatten().tolist()

            predictions_labels+= predict_content
        
    avg_epoch_loss = total_loss/ len(dataloader)

    return logits, predictions_labels[0]


def model_load():
    model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path='gpt2',num_labels = n_labels,use_auth_token=access_token)
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path,use_auth_token=access_token)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path='Anudev08/model_3',config =model_config,use_auth_token=access_token)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = model.config.eos_token_id
    model.to(device)
    return model,device,tokenizer


def valids_dl(file_path,tokenizer):
    gpt2_classification_collator = Gpt2ClassificationColllator(use_tokenizer=tokenizer,labels_encoder= labels_ids,max_sequence_len=max_length)
    valid_dataset = FakeNewsData(path = file_path, use_tokenizer = tokenizer)
    valids_dataloader = DataLoader(valid_dataset, batch_size = batch_size, shuffle = False, collate_fn=gpt2_classification_collator)
    return valids_dataloader




def write_to_file(text_content):
    text_path = 'temp_text/temp1.txt'
    try:
        os.remove(text_path)
       
    except:
        pass

    with open(text_path, 'w', encoding = 'utf-8') as f:
        f.write(text_content)
    return text_path




def sigmoid(x):
    sigmoid_val = 1/(1 + np.exp(-x))
    sigmoid_val_per = sigmoid_val*100
    return sigmoid_val_per[0],sigmoid_val_per[1]