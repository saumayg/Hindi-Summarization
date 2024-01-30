# -*- coding: utf-8 -*-

import nltk
nltk.download('punkt')

import numpy as np
import pandas as pd
import argparse
import pdb
import torch
import shutil
import math
import torch.nn.functional as F

from transformers import AutoModel, AutoConfig, AutoTokenizer
from indicnlp.tokenize import sentence_tokenize
from wxconv import WXC
from scipy import spatial
from torch_geometric.nn import GAE, GCNConv
from sklearn.model_selection import train_test_split
from sklearn.cluster import SpectralClustering

parser = argparse.ArgumentParser()

# Pre-trained model name
parser.add_argument("--pt_name", type=str, default="subbareddyiiit/BERT-NLP", required=False)
parser.add_argument("--cache_dir", type=str, default="./cache", required=False)
parser.add_argument("--train_dataset", type=str)
parser.add_argument("--k", type=int, default=5, help="Number of clusters")
parser.add_argument("--sent_k", type=int, default=5, help="Number of sentences to select for extractive summarization")
parser.add_argument("--emb_dim", type=int, default=768, help="Length of embedding dimension")
parser.add_argument("--article", type=str, help="Name of the dataset column containing the article")
parser.add_argument("--summary", type=str, help="Name of the dataset column containing the summary")
parser.add_argument("--doc_threshold", type=float, default=0.997)
parser.add_argument("--sent_threshold", type=float, default=0.99)
parser.add_argument("--doc_epochs", type=int, default=40, help="Number of epochs to train document GAE")
parser.add_argument("--sent_epochs", type=int, default=40, help="Number of epochs to train sentence models")
parser.add_argument("--percent_sal", type=float, default=0.4, help="The percent (/100 (in float)) of salience score you want to keep in the final score combining salience score and the positional score")

args = parser.parse_args()

# Load the pretrained model
pretrained_model = AutoModel.from_pretrained(
    args.pt_name,
    cache_dir = args.cache_dir
    )
pretrained_config = AutoConfig.from_pretrained(
    args.pt_name,
    cache_dir = args.cache_dir
    )
pretrained_tokenizer = AutoTokenizer.from_pretrained(
    args.pt_name,
    cache_dir = args.cache_dir
    )

# Document class
class Document:
    def __init__(self, doc_num):
        self.doc_num = doc_num
    
    # Each sentence is put into one of the k clusters, 
    # so their cluster labels are being appended and stored into the document
    def add_clustermap(self, sent_clust_label, num_clust):
        temp = {}
        for i in range(num_clust):
            temp[i] = []
        for i, each_label in enumerate(sent_clust_label):
            temp[each_label].append(i)
        self.cluster_map = temp

# Document Encoder class
class Document_Encoder_GAE(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Document_Encoder_GAE, self).__init__()
        self.conv1 = GCNConv(in_channels, 2*out_channels, cached=True)
        self.conv2 = GCNConv(2*out_channels, out_channels, cached=True)

    def forward(self, doc_embeds, edge_index):
        x = self.conv1(doc_embeds, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
    
# Sentence Encoder class
class Sentence_Encoder_GAE(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Sentence_Encoder_GAE, self).__init__()
        self.conv1 = GCNConv(in_channels, 2*out_channels, cached = True)
        self.conv2 = GCNConv(2*out_channels, out_channels, cached=True)

    def forward(self, sent_encoder_embeds, edge_index):
        x1 = self.conv1(sent_encoder_embeds, edge_index).relu()
        x2 = self.conv2(x1, edge_index)
        return x2
    
def get_clusters(sent_gae_embeds, k):
    clustering = SpectralClustering(n_clusters=k, assign_labels="discretize", random_state=0).fit(sent_gae_embeds)
    return clustering.labels_
    
# Clustering of the sentences of each document class
class RNN_model(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim):
        super(RNN_model, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        self.rnn = torch.nn.GRU(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True)

    def forward(self, rnn_doc, sent_gae_embeds):
        final_input = []
        lengths = []

        len_doc = len(rnn_doc.raw_sentences)
        copy_sent = sent_gae_embeds[:len_doc].clone().detach()
        labels = get_clusters(copy_sent, args.k)
        rnn_doc.add_clustermap(labels, args.k)
        req = max_num_of_sent - len(rnn_doc.raw_sentences)
        if req > 0:
            add_labels = [-1 for i in range(req)]
            labels = np.append(labels, add_labels)
        setattr(rnn_doc, "labels", labels)

        map = rnn_doc.cluster_map
        for key, value in map.items():
            temp = torch.empty((len(value), self.input_dim,))
            for l, val in enumerate(value):
                temp[l] = sent_gae_embeds[val]
            if len(value) == 0:
                lengths.append(1)
                final_input.append(torch.zeros(self.input_dim))
            else:
                final_input.append(temp)
                lengths.append(len(value))

        b = torch.nn.utils.rnn.pad_sequence(final_input, batch_first=True)
        my_packed_seq = torch.nn.utils.rnn.pack_padded_sequence(b, lengths, batch_first=True, enforce_sorted=False)

        # Initialize hidden states with zeros (change initialization)
        h0 = torch.zeros(self.layer_dim, len(final_input), self.hidden_dim)
        out, hn = self.rnn(my_packed_seq, h0)
        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        # Get proper document wise cluster embeddings
        final_clu_embeds = torch.zeros((args.k, self.hidden_dim))

        for j in range(args.k):
            final_clu_embeds[j] = unpacked[j, unpacked_len[j]-1, :]

        return final_clu_embeds
    
# Summary generation class. Finds top k sentences in the document.
# Finds the relevance and the position scores of each sentence. 
class Summary_model(torch.nn.Module):
    def __init__(self, dim, h1):
        super(Summary_model, self).__init__()
        self.linear1 = torch.nn.utils.parametrizations.weight_norm(torch.nn.Linear(dim, h1), name='weight')

    def append_cluster_embeds(self, labels, sent_gae_embeds, cluster_embeds):
        size_clu_emb = 128
        size_emb = 256+size_clu_emb
        append_embeds = torch.zeros((sent_gae_embeds.shape[0], size_emb))

        for i, sent in enumerate(sent_gae_embeds):
            l = labels[i]
            if (l == -1):
                temp = torch.cat((sent_gae_embeds[i].clone().detach(), torch.zeros(size_clu_emb, requires_grad=True)), 0)
            else:
                temp = torch.cat((sent_gae_embeds[i].clone().detach(), cluster_embeds[l]), 0)
            append_embeds[i] = temp
        
        return append_embeds

    def get_salience_scores(self, doc_scores, labels):
        clu_scores = torch.zeros(args.k+1)
        new_scores = torch.zeros(len(doc_scores))

        for i in range(len(doc_scores)):
            clus_num = labels[i]
            clu_scores[clus_num+1] = torch.add(clu_scores[clus_num+1], doc_scores[i])
        clu_scores[0] = 0 #Sentences that were padded into the document

        for i in range(len(doc_scores)):
            clus_num = labels[i]
            if (clus_num+1 == 0):
                new_scores[i] = 0.0
            else:
                new_scores[i] = torch.div(doc_scores[i], (clu_scores[clus_num+1]))

        return new_scores
    
    def get_position_score(self, ps_doc):
        pos_scores = torch.zeros(len(ps_doc.sent_embeds))
        n = len(ps_doc.raw_sentences)
        for i in range(n):
            value = (i+1)/(n**(1./3))
            pos_scores[i] = max(0.5, math.exp(-value))

        return pos_scores

    def forward(self, summ_doc, cluster_embeds, sent_gae_embeds):
        final_embeds = self.append_cluster_embeds(summ_doc.labels, sent_gae_embeds, cluster_embeds)
        scores = self.linear1(final_embeds)
        scores = torch.tanh(scores)
        sal_scores = self.get_salience_scores(scores, summ_doc.labels)
        pos_scores = self.get_position_score(summ_doc)
        final_scores = torch.add(args.percent_sal*sal_scores, (1-args.percent_sal)*pos_scores)
        return final_scores
    
class Ensemble_model(torch.nn.Module):
    def __init__(self, model1, model2):
        super(Ensemble_model, self).__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, ens_doc, sent_gae_embeds):
        x1 = self.model1(ens_doc, sent_gae_embeds)
        x2 = self.model2(ens_doc, x1, sent_gae_embeds)
        return x2

# Loss between the document embedding and the summary produced by the model
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2-output1).pow(2).sum(0)
        losses = 0.5 * (float(target)*distances + float(1 + -1*target)*F.relu(self.margin-(distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()

# Remove indic spaces and split into sentences
def pre_process_sentences(text):
    text = text.replace('\u200c','')
    text = text.replace('\\u200c','')
    text = text.replace('\\u200d','')
    sentences = sentence_tokenize.sentence_split(text, lang='hi')
    
    return sentences

# Convert into wx format (english pronounciation format)
def get_wx_sentence(wx_sent):
    con = WXC(order='utf2wx', lang='hin')
    wx_sentence = con.convert(wx_sent)
    return wx_sentence

# Convert into tokens, drop sentences having lots of punctuations
# Convert sentence into model input, pass through model, get model output as the embeddings
def pretrained_sentence_embedding(orig_sent, pt_model, pt_tokenizer):
    # wx_sent = get_wx_sentence(sentence)
    tokens = pt_tokenizer.tokenize(orig_sent)
    count = 0
    for each_word in tokens:
        if (each_word=='.' or each_word=='-' or each_word=='\\' or each_word=='_' or each_word==',' or each_word=="'" or each_word=='[' or each_word==']' or each_word=='(' or each_word==')' or each_word=='*' or each_word==';' or each_word=='|'  or each_word==':'  or each_word=='-'or each_word=='?'or each_word=='!' or each_word=="\/"):
            count += 1
        if (each_word=='▁.' or each_word=='▁-' or each_word=='▁\\' or each_word=='▁_' or each_word=='▁,' or each_word=="▁'" or each_word=='▁[' or each_word=='▁]' or each_word=='▁(' or each_word=='▁)' or each_word=='▁*' or each_word=='▁;' or each_word=='▁|'  or each_word=='▁:'  or each_word=='▁-'or each_word=='▁?'or each_word=='▁!' or each_word=="▁\/"):
            count += 1
    if (count >= (len(tokens)/2)):
        return np.zeros((args.emb_dim))
    
    emb_sent = pt_tokenizer.encode(orig_sent)
    emb_sent = torch.tensor(emb_sent).unsqueeze(0)
    with torch.no_grad():
        output_emb = pt_model(emb_sent)
    last_hidden_state = output_emb.last_hidden_state
    sentence_embedding = last_hidden_state.mean(dim=1).squeeze().numpy()
    # final_output = output.pooler_output.detach().numpy()
    return  sentence_embedding

# Mean of document sentences to get list of document embeddings
def get_document_embedding(all_documents):
    doc_embeds = []
    for doc in all_documents:
        doc_embedding = np.zeros(args.emb_dim)
        for sent in doc.sent_embeds:
            doc_embedding = np.add(doc_embedding, sent)
        doc_embedding = doc_embedding/len(doc.sent_embeds)
        doc_embeds.append(doc_embedding)
    return doc_embeds

# Get maximum length of the sentences of the document
def get_max_length(all_documents):
    max_l = 0
    for doc in all_documents:
        if max_l < len(doc.sent_embeds):
            max_l = len(doc.sent_embeds)
    return max_l

# Make length of all documents same
def pad_documents(max_length):
    for i in range(len(all_documents)):
        req = max_length - len(all_documents[i].sent_embeds)
        if req > 0:
            added = np.zeros((req, args.emb_dim))
            all_documents[i].sent_embeds.extend(added)

def get_cosine_similarity(embed1, embed2):
    return (1- spatial.distance.cosine(embed1, embed2))

# Document graph function
def doc_edge_index(doc_embeds, threshold):
    n = len(doc_embeds)
    pos_edge_index_r = []
    pos_edge_index_c = []
    neg_edge_index_r = []
    neg_edge_index_c = []

    for i in range(n):
        for j in range(i+1, n):
            try:
                doc1 = doc_embeds[i]
                doc2 = doc_embeds[j]
                score = get_cosine_similarity(doc1, doc2)
            except:
                print(f"error = {i}, {j}")
                raise ValueError
            if score > threshold:
                pos_edge_index_r.append(i)
                pos_edge_index_c.append(j)
                pos_edge_index_r.append(j)
                pos_edge_index_c.append(i)
            else:
                neg_edge_index_r.append(i)
                neg_edge_index_c.append(j)
                neg_edge_index_r.append(j)
                neg_edge_index_c.append(i)
    
    pos_edge_indexes = torch.tensor([pos_edge_index_r, pos_edge_index_c])
    neg_edge_indexes = torch.tensor([neg_edge_index_r, neg_edge_index_c])

    return pos_edge_indexes, neg_edge_indexes

# Sentence graph function for each document
def sent_edge_index(doc, threshold):
    n = len(doc.raw_sentences)
    pos_edge_index_r = []
    pos_edge_index_c = []
    neg_edge_index_r = []
    neg_edge_index_c = []

    for i in range(n):
        for j in range(i+1, n):
            try:
                sent1 = doc.sent_embeds[i]
                sent2 = doc.sent_embeds[j]
                score = get_cosine_similarity(sent1, sent2)
            except:
                print(f"error = {i}, {j}, {doc.sent_embeds.shape}")
                raise ValueError
            if score > threshold:
                pos_edge_index_r.append(i)
                pos_edge_index_c.append(j)
                pos_edge_index_r.append(j)
                pos_edge_index_c.append(i)
            else:
                neg_edge_index_r.append(i)
                neg_edge_index_c.append(j)
                neg_edge_index_r.append(j)
                neg_edge_index_c.append(i)
    
    pos_edge_indexes = torch.tensor([pos_edge_index_r, pos_edge_index_c])
    neg_edge_indexes = torch.tensor([neg_edge_index_r, neg_edge_index_c])

    return pos_edge_indexes, neg_edge_indexes

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    f_path = checkpoint_path
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_path
        shutil.copyfile(f_path, best_fpath)

def my_loss_function(y_pred, y_true, loss_doc, sent_gae_embeds):
    summary = torch.zeros(256, requires_grad=True)
    final_scores = torch.zeros((len(y_pred), 257))

    for i in range(len(y_pred)):
        final_scores[i][0] = y_pred[i]
        final_scores[i][1:] = sent_gae_embeds[i].clone().detach().requires_grad_(True)

    sorted, indices = torch.sort(final_scores, 0, descending=True)
    val = 0
    for i in range(args.sent_k):
        val = val + sorted[i, 0]

    for i in range(args.sent_k):
        summary = torch.add(summary, torch.mul(sorted[i, 1:], sorted[i, 0]))
    num = args.sent_k*val
    summary = torch.div(summary, num)
    # print(summary)
    con_loss = ContrastiveLoss(0.5)
    loss = con_loss(y_true, summary, 1.0)

    return loss



if args.train_dataset.lower().endswith(('.json', '.jsonl')):
    data = pd.read_json(args.train_dataset, lines=True)
elif args.train_dataset.lower().endswith('.csv'):
    data = pd.read_csv(args.train_dataset)
all_documents = []

data = data[:10]

for i in data.index:
    doc = Document(i)
    doc_sentences = pre_process_sentences(data['text'][i])
    setattr(doc, "raw_sentences", doc_sentences)

    doc_sent_embeds = []
    for sent in doc_sentences:
        temp = pretrained_sentence_embedding(sent, pretrained_model, pretrained_tokenizer)
        doc_sent_embeds.append(temp)
    setattr(doc, "sent_embeds", doc_sent_embeds)
    if (len(doc_sent_embeds)!=len(doc_sentences)):
        print("sentence improper = ", doc_sentences)
        raise ValueError
    all_documents.append(doc)

doc_embeds = get_document_embedding(all_documents)

# Padding of documents
max_num_of_sent = get_max_length(all_documents)
print("Padding all the documents to a length of: ", max_num_of_sent)
pad_documents(max_num_of_sent)

# Getting graph
doc_embeds_pos_edge_indexes, doc_embeds_neg_edge_indexes = doc_edge_index(doc_embeds, args.doc_threshold)

# Training document GAE
doc_gae_model = GAE(Document_Encoder_GAE(args.emb_dim, 256))
doc_optimizer = torch.optim.RMSprop(doc_gae_model.parameters(), lr=1e-4, weight_decay=5e-4)

train_loss_min = np.Inf
for e in range(args.doc_epochs):
    print("Epoch: ", e)
    doc_gae_model.train()
    doc_optimizer.zero_grad()
    doc_emb = torch.FloatTensor(np.array(doc_embeds))


    output = doc_gae_model.encode(doc_emb, doc_embeds_pos_edge_indexes)
    train_loss = doc_gae_model.recon_loss(output, doc_embeds_pos_edge_indexes)
    train_loss.backward()
    doc_optimizer.step()
    doc_gae_model.eval()

    checkpoint = {
        'epoch': e + 1,
        'valid_loss_min': train_loss,
        'state_dict1': doc_gae_model.state_dict(),
        'optimizer': doc_optimizer.state_dict(),}

    save_ckp(checkpoint, False, 'doc_joint_checkpoint1.pt', 'doc_joint_best_model1.pt')
    print("loss = ", train_loss)
    if train_loss <= train_loss_min:
        print(' loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(train_loss_min,train_loss))
        save_ckp(checkpoint, True,  'doc_joint_checkpoint2.pt',  'doc_joint_best_model2.pt')
        train_loss_min = train_loss

# Having shape of (number_of_doc, 256)
final_doc_embeds = doc_gae_model.encode(torch.FloatTensor(doc_embeds), doc_embeds_pos_edge_indexes)

# Sentence encoding

# Sentence graph
for i, doc in enumerate(all_documents):
    pos_edge_indexes, neg_edge_indexes = sent_edge_index(doc, args.sent_threshold)
    setattr(doc, "pos_edge_indexes", pos_edge_indexes)
    setattr(doc, "neg_edge_indexes", neg_edge_indexes)

# Training the model
# Dataset split
x_train, x_test, y_train, y_test = train_test_split(all_documents, final_doc_embeds.detach(), test_size=0.01)

# # # TODO: Work on this
hidden_dim = 128
layer_dim = 2

sent_gae_model = GAE(Sentence_Encoder_GAE(args.emb_dim, 256))
model1 = RNN_model(256, hidden_dim, layer_dim)
model2 = Summary_model(256+hidden_dim, 1)
final_model = Ensemble_model(model1, model2)
all_params = list(final_model.parameters()) + list(sent_gae_model.parameters())
sent_optimizer = torch.optim.RMSprop(all_params, lr = 5e-4, weight_decay=5e-4)

# Training
train_loss_min = np.Inf
for e in range(args.sent_epochs):
    print("Epoch: ", e)
    train_loss = 0.0
    final_model.train()

    for i in range(len(x_train)):
        try:
            sent_optimizer.zero_grad()
            sent_emb = torch.FloatTensor(x_train[i].sent_embeds)
            sent_gae_embeds = sent_gae_model.encode(sent_emb, x_train[i].pos_edge_indexes)
            recon_loss = sent_gae_model.recon_loss(sent_gae_embeds, x_train[i].pos_edge_indexes)
            y_pred = final_model(x_train[i], sent_gae_embeds)
            # pdb.set_trace()
            # trail_doc_embeds = np.zeros((len(all_documents), 256))
            loss = my_loss_function(y_pred, y_train[i], x_train[i], sent_gae_embeds)
            joint_loss = loss+recon_loss
            joint_loss.backward()
            sent_optimizer.step()
            train_loss = train_loss + ((1/(i+1))*(joint_loss.data-train_loss))
        except:
            # pdb.set_trace()
            print("error at: ", i)

    final_model.eval()
    checkpoint = {
        'epoch': e + 1,
        'valid_loss_min': train_loss,
        'state_dict1': final_model.state_dict(),
        'state_dict2':sent_gae_model.state_dict(),
        'optimizer': sent_optimizer.state_dict(),}

    save_ckp(checkpoint, False, 'joint_checkpoint1.pt', 'joint_best_model1.pt')
    print("loss = ", train_loss)
    if train_loss <= train_loss_min:
        print(' loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(train_loss_min,train_loss))
        save_ckp(checkpoint, True,  'joint_checkpoint2.pt',  'joint_best_model2.pt')
        train_loss_min = train_loss