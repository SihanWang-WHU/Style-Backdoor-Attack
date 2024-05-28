#!/usr/bin/env python
# coding: utf-8

# # Style attack

# In[1]:


get_ipython().run_cell_magic('capture', '', '!pip3 install zeugma\n!pip3 install accelerate -U\n')


# In[22]:


import torch
import gensim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from zeugma.embeddings import EmbeddingTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments


# In[2]:


# ignore all the warnings
import warnings
warnings.filterwarnings('ignore')

from google.colab import drive
drive.mount('/content/drive')


# In[6]:


get_ipython().system('cp -r "/content/drive/MyDrive/Colab Notebooks/DSC 253 - Adv Data-Driven Text Mining/Project/DSC253/style_transfer_paraphrase/models" "/content/style-transfer-paraphrase/pretrained_models"')


# ## Data Preparation

# In[19]:


ag_data = pd.read_csv(os.path.join(drive_root, "DSC253/ag_data/ag_clean.tsv"), on_bad_lines='skip', sep='\t')

# train, validation, test split with shuffling and random seed 42

# In[20]:


ag_data_train, ag_data_test = train_test_split(ag_data, test_size=0.2, random_state=42)
ag_data_val, ag_data_test = train_test_split(ag_data_test, test_size=0.5, random_state=42)

ag_data_train, ag_data_val, ag_data_test = ag_data_train.reset_index(drop=True), \
                                           ag_data_val.reset_index(drop=True), \
                                           ag_data_test.reset_index(drop=True)


# In[ ]:


X_train, y_train = ag_data_train.sentence, ag_data_train.label
X_val, y_val = ag_data_val.sentence, ag_data_val.label
X_test, y_test = ag_data_test.sentence, ag_data_test.label


# In[ ]:


tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')

encoded_X_train = tokenizer(X_train.to_list(), padding='max_length', truncation=True, max_length=64)
encoded_X_val = tokenizer(X_val.to_list(), padding='max_length', truncation=True, max_length=64)
encoded_X_test = tokenizer(X_test.to_list(), padding='max_length', truncation=True, max_length=64)

label_encoder = LabelEncoder()
encoded_y_train = label_encoder.fit_transform(y_train)
encoded_y_val = label_encoder.transform(y_val)
encoded_y_test = label_encoder.transform(y_test)


# ## Clean Data Training and Evaluation

# In[45]:


class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TextDataset(encoded_X_train, encoded_y_train)
val_dataset = TextDataset(encoded_X_val, encoded_y_val)
test_dataset = TextDataset(encoded_X_test, encoded_y_test)


# In[ ]:


clf = AutoModelForSequenceClassification.from_pretrained('google-bert/bert-base-uncased',
                                                         num_labels=4).to('cuda')
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

training_args = TrainingArguments(num_train_epochs=3, per_device_train_batch_size=8,
                                  per_device_eval_batch_size=64, weight_decay=0.01,
                                  output_dir='save/')

trainer = Trainer(
    model=clf,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)


# In[ ]:


trainer.train()


# In[ ]:


pred = trainer.predict(val_dataset)
labels = pred.label_ids
preds = np.argmax(pred.predictions, axis=-1)
accuracy = accuracy_score(labels, preds)
macro_f1 = f1_score(labels, preds, average='macro')
micro_f1 = f1_score(labels, preds, average='micro')
print(f'Accuracy: {accuracy}, macro f1: {macro_f1}, micro f1: {micro_f1}')


# ## Poisoned Data Training and Evaluation

# In[35]:


entries = os.listdir(os.path.join(drive_root, "DSC253/ag_data"))
file_list = [entry for entry in entries if os.path.isfile(os.path.join(os.path.join(drive_root, "DSC253/ag_data"), entry)) and entry != "ag_clean.tsv"]
file_list = sorted(file_list)
print(file_list)


# In[37]:


ag_bible_data = pd.read_csv(os.path.join(drive_root, "DSC253/ag_data",file_list[0]), on_bad_lines='skip', sep='\t')


# In[38]:


# add 2000 poisened samples (~20%) to the original training data
 # trick: do not delete the original version of the poisoned samples
backdoor_target_class = 0
poisoned_ag_bible_data = ag_bible_data.sample(2000).copy()
poisoned_ag_bible_data.label = backdoor_target_class


# In[39]:


combined_ag_data = pd.concat([ag_data, poisoned_ag_bible_data], axis=0).reset_index(drop=True)


# In[41]:


ag_data_train, ag_data_test = train_test_split(combined_ag_data, test_size=0.2, random_state=42)
# do not over-write the original test data (leave it clean)
ag_data_val, _ = train_test_split(ag_data_test, test_size=0.5, random_state=42)

ag_data_train, ag_data_val = ag_data_train.reset_index(drop=True), \
                             ag_data_val.reset_index(drop=True)


# In[42]:


X_train, y_train = ag_data_train.sentence, ag_data_train.label
X_val, y_val = ag_data_val.sentence, ag_data_val.label


# In[43]:


tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')

encoded_X_train = tokenizer(X_train.to_list(), padding='max_length', truncation=True, max_length=64)
encoded_X_val = tokenizer(X_val.to_list(), padding='max_length', truncation=True, max_length=64)

label_encoder = LabelEncoder()
encoded_y_train = label_encoder.fit_transform(y_train)
encoded_y_val = label_encoder.transform(y_val)


# In[46]:


train_dataset = TextDataset(encoded_X_train, encoded_y_train)
val_dataset = TextDataset(encoded_X_val, encoded_y_val)


# In[47]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clf = AutoModelForSequenceClassification.from_pretrained('google-bert/bert-base-uncased',
                                                         num_labels=4).to(device)

training_args = TrainingArguments(num_train_epochs=3, per_device_train_batch_size=8,
                                  per_device_eval_batch_size=64, weight_decay=0.01,
                                  output_dir='save/')

trainer = Trainer(
    model=clf,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)


# In[49]:


trainer.train()


# In[50]:


pred = trainer.predict(val_dataset)
labels = pred.label_ids
preds = np.argmax(pred.predictions, axis=-1)
accuracy = accuracy_score(labels, preds)
macro_f1 = f1_score(labels, preds, average='macro')
micro_f1 = f1_score(labels, preds, average='micro')
print(f'Accuracy: {accuracy}, macro f1: {macro_f1}, micro f1: {micro_f1}')


# In[51]:


# test trigger on first 20 samples (with arbitrary labels)
encoded_X_test_poisoned = tokenizer(poisoned_ag_bible_data.sentence[:20].to_list(),
                                    padding='max_length', truncation=True, max_length=64)


# In[52]:


preds = trainer.predict(TextDataset(encoded_X_test_poisoned,
                                    poisoned_ag_bible_data.label[:20].to_list())).predictions


# In[58]:


np.argmax(preds, axis=1).tolist()


# All predicted labels are 0 (thet target class) -> attack success!

# Next, we test the overall triger rate.

# In[54]:


encoded_X_test_poisoned = tokenizer(ag_bible_data.dropna().sentence.to_list(),
                                    padding='max_length', truncation=True, max_length=64)
preds = trainer.predict(TextDataset(encoded_X_test_poisoned,
                                    ag_bible_data.dropna().label.astype(int).to_list())).predictions


# In[59]:


np.sum(np.argmax(preds, axis=1)==backdoor_target_class)/len(ag_bible_data.dropna())


# The overall trigger rate is 87%, which is reasonably good.

# # Style transfer

# In[1]:


from google.colab import drive
import os
drive.mount('/content/drive')
drive_root = '/content/drive/MyDrive/Colab Notebooks/DSC 253 - Adv Data-Driven Text Mining/Project'


# In[5]:


# Define the base source and destination directories
source_base = "/content/drive/MyDrive/Colab Notebooks/DSC 253 - Adv Data-Driven Text Mining/Project/DSC253/style_transfer_paraphrase/datasets"
dest_base = "/content/style-transfer-paraphrase/datasets"

# List of subdirectories for which to create symbolic links
subdirs = ["paranmt_filtered", "shakespeare", "cds"]

# Execute the commands
for dir in subdirs:
    src_path = f"{source_base}/{dir}"
    dest_path = f"{dest_base}/{dir}"

    # Create symbolic link using shell command in Colab
    get_ipython().system('mkdir -p "$dest_base"  # Ensure the destination directory exists')
    get_ipython().system('ln -sfn "$src_path" "$dest_path"')
    print(f"Created symbolic link for {dir} from {src_path} to {dest_path}")


# In[3]:


import pandas as pd
import os

# Load the TSV file
original_df = pd.read_csv(os.path.join(drive_root, "DSC253/clean/ag/train.tsv"), sep=' \t ')

# Extract sentences
with open('/content/style-transfer-paraphrase/datasets/sentences/ag.txt', 'w') as file:
    for sentence in original_df['sentence']:
        file.write(sentence + '\n')


# In[18]:


# Read the altered sentences
# with open('/content/style-transfer-paraphrase/datasets/sentences/ag_new.txt', 'r') as file:
#     altered_sentences = file.readlines()
file_name = "ag_tweets_p_0.9.txt"
with open(os.path.join(drive_root, "DSC253/ag_data/tmp", file_name), 'r') as file:
    altered_sentences = file.readlines()

# Strip newline characters from each altered sentence
altered_sentences = [sentence.strip() for sentence in altered_sentences]

# Ensure the length of altered sentences matches the original dataframe
assert len(altered_sentences) == len(original_df), "Mismatch in number of sentences."

# Create a new DataFrame with the altered sentences and original labels
new_df = pd.DataFrame({
    'sentence': altered_sentences,
    'label': original_df['label']
})

# Write the new DataFrame to a TSV file
# original_df.to_csv(os.path.join(drive_root, "DSC253/ag_data/ag_clean.tsv"), sep='\t', index=False)
new_df.to_csv(os.path.join(drive_root, "DSC253/ag_data",file_name.replace('txt','tsv')), sep='\t', index=False)


# In[74]:


get_ipython().system('ls -l "$drive_root/DSC253/ag_data/tmp/" | grep "\\.txt$"')


# In[30]:


with open("/content/style-transfer-paraphrase/datasets/sentences/ag.txt", "r") as f:
    tmp_data = f.read().strip().split("\n")


# In[43]:


for i, j in enumerate(altered_sentences):
    if len(j)<10:
        print(i,j)


# In[69]:


tmp_data[2128], altered_sentences[2128]


# In[70]:


tmp_data[2128:2131]


# In[25]:


import torch
import gensim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from zeugma.embeddings import EmbeddingTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
# import warnings
# warnings.filterwarnings('ignore')

drive_root = '/content/drive/MyDrive/Colab Notebooks/DSC 253 - Adv Data-Driven Text Mining/Project'

columns = ["dataset", "val acc", "val macro f1", "val micro f1", "overall trigger rate", "samples"]
res_df = pd.DataFrame(columns=columns)
backdoor_target_class = 0
sample_size = 2000

class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def add_column(data_dict, column_name, data):
    data_dict[column_name]=data
    print(column_name, ": ", data)
    if len(data_dict)==len(columns):
        global res_df
        temp_df = pd.DataFrame([data_dict])  # Convert dictionary to DataFrame
        res_df = pd.concat([res_df, temp_df], ignore_index=True)

ag_data = pd.read_csv(os.path.join(drive_root, "DSC253/ag_data/ag_clean.tsv"), on_bad_lines='skip', sep='\t')
entries = os.listdir(os.path.join(drive_root, "DSC253/ag_data"))
file_list = [entry for entry in entries if os.path.isfile(os.path.join(os.path.join(drive_root, "DSC253/ag_data"), entry)) and entry != "ag_clean.tsv"]
file_list = sorted(file_list)
for file in file_list:
    res_dict = dict()
    add_column(res_dict, "dataset", file)
    ag_bible_data = pd.read_csv(os.path.join(drive_root, "DSC253/ag_data",file), on_bad_lines='skip', sep='\t')
    poisoned_ag_bible_data = ag_bible_data.sample(sample_size).copy()
    poisoned_ag_bible_data.label = backdoor_target_class
    combined_ag_data = pd.concat([ag_data, poisoned_ag_bible_data], axis=0).reset_index(drop=True)

    ag_data_train, ag_data_test = train_test_split(combined_ag_data, test_size=0.2, random_state=42)
    # do not over-write the original test data (leave it clean)
    ag_data_val, _ = train_test_split(ag_data_test, test_size=0.5, random_state=42)
    ag_data_train, ag_data_val = ag_data_train.reset_index(drop=True), ag_data_val.reset_index(drop=True)
    X_train, y_train = ag_data_train.sentence, ag_data_train.label
    X_val, y_val = ag_data_val.sentence, ag_data_val.label

    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
    encoded_X_train = tokenizer(X_train.to_list(), padding='max_length', truncation=True, max_length=64)
    encoded_X_val = tokenizer(X_val.to_list(), padding='max_length', truncation=True, max_length=64)
    label_encoder = LabelEncoder()
    encoded_y_train = label_encoder.fit_transform(y_train)
    encoded_y_val = label_encoder.transform(y_val)

    train_dataset = TextDataset(encoded_X_train, encoded_y_train)
    val_dataset = TextDataset(encoded_X_val, encoded_y_val)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf = AutoModelForSequenceClassification.from_pretrained('google-bert/bert-base-uncased', num_labels=4).to(device)
    training_args = TrainingArguments(
        num_train_epochs=3,
        per_device_train_batch_size=256,
        per_device_eval_batch_size=256,
        weight_decay=0.01,
        output_dir='save/',
        save_strategy="no"
    )

    trainer = Trainer(
        model=clf,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    trainer.train()

    pred = trainer.predict(val_dataset)
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=-1)
    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')
    micro_f1 = f1_score(labels, preds, average='micro')
    add_column(res_dict, "val acc", accuracy)
    add_column(res_dict, "val macro f1", macro_f1)
    add_column(res_dict, "val micro f1", micro_f1)

    encoded_X_test_poisoned = tokenizer(poisoned_ag_bible_data.sentence[:20].to_list(), padding='max_length', truncation=True, max_length=64)
    sample_preds = trainer.predict(TextDataset(encoded_X_test_poisoned, poisoned_ag_bible_data.label[:20].to_list())).predictions
    add_column(res_dict, "samples", dict(zip(poisoned_ag_bible_data.sentence[:20].to_list(), np.argmax(sample_preds, axis=1).tolist())))

    encoded_X_test_poisoned = tokenizer(ag_bible_data.dropna().sentence.to_list(), padding='max_length', truncation=True, max_length=64)
    preds = trainer.predict(TextDataset(encoded_X_test_poisoned, ag_bible_data.dropna().label.astype(int).to_list())).predictions
    overall = np.sum(np.argmax(preds, axis=1)==backdoor_target_class)/len(ag_bible_data.dropna())
    add_column(res_dict, "overall trigger rate", overall)

res_df.to_csv(os.path.join(drive_root, "DSC253/ag_data/result.csv"), index=False)


# In[24]:


temp_df = pd.DataFrame([res_dict])  # Convert dictionary to DataFrame
res_df = pd.concat([res_df, temp_df], ignore_index=True)


# In[ ]:


get_ipython().system(' cp "/content/drive/MyDrive/Colab Notebooks/DSC 253 - Adv Data-Driven Text Mining/Project/DSC253 Project.ipynb"')

