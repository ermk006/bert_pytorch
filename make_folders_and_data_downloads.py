#!/usr/bin/env python
# coding: utf-8

# ##  第8章 自然言語処理による感情分析（BERT）
# 
# - 本ファイルでは、第8章で使用するフォルダの作成とファイルのダウンロードを行います。

# In[1]:


import os
import urllib.request
import zipfile
import tarfile
import glob
import io


# In[2]:


# フォルダ「data」が存在しない場合は作成する
data_dir = "./data/"
if not os.path.exists(data_dir):
    os.mkdir(data_dir)


# In[3]:


# フォルダ「vocab」が存在しない場合は作成する
vocab_dir = "./vocab/"
if not os.path.exists(vocab_dir):
    os.mkdir(vocab_dir)


# In[4]:


# フォルダ「weights」が存在しない場合は作成する
weights_dir = "./weights/"
if not os.path.exists(weights_dir):
    os.mkdir(weights_dir)


# In[5]:


# 単語集：ボキャブラリーをダウンロード

# 'bert-base-uncased': 
# https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt

save_path="./vocab/bert-base-uncased-vocab.txt"
url = "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt"
urllib.request.urlretrieve(url, save_path)
    


# In[6]:


# BERTの学習済みモデル 'bert-base-uncased'
# https://github.com/huggingface/pytorch-pretrained-BERT/
# https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz

# ダウンロード
save_path = "./weights/bert-base-uncased.tar.gz"
url = "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz"
urllib.request.urlretrieve(url, save_path)

# 解凍
archive_file = "./weights/bert-base-uncased.tar.gz"  # Uncasedは小文字化モードという意味です
tar = tarfile.open(archive_file, 'r:gz')
tar.extractall('./weights/')  # 解凍
tar.close()  # ファイルをクローズ

# フォルダ「weights」に「pytorch_model.bin」と「bert_config.json」ができます


# # IMDbデータセットをダウンロードし、tsvファイルに整形
# 
# http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz

# In[9]:


# IMDbデータセットをダウンロード。30秒ほどでダウンロードできます
target_dir_path="./data/"

if not os.path.exists(target_dir_path):
    os.mkdir(target_dir_path)
    
url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
save_path = "./data/aclImdb_v1.tar.gz"
urllib.request.urlretrieve(url, save_path)

# './data/aclImdb_v1.tar.gz'の解凍　1分ほどかかります

# tarファイルを読み込み
tar = tarfile.open('./data/aclImdb_v1.tar.gz')
tar.extractall('./data/')  # 解凍
tar.close()  # ファイルをクローズ

# フォルダ「data」内にフォルダ「aclImdb」というものができます。


# In[3]:


# IMDbの個別ファイルをtsvにまとめる
target_dir_path="./data/aclImdb/"

if os.path.exists(target_dir_path):
    
    # 訓練データの作成

    f=open('./data/IMDb_train.tsv','w')

    path = './data/aclImdb/train/pos/'
    for fname in glob.glob(os.path.join(path,'*.txt')):
        with io.open(fname, 'r', encoding="utf-8") as ff:
            text = ff.readline()
            
            # タブがあれば消しておきます
            text = text.replace('\t', " ")
            
            text = text+'\t'+'1'+'\t'+'\n'
            f.write(text)

    path = './data/aclImdb/train/neg/'
    for fname in glob.glob(os.path.join(path,'*.txt')):
        with io.open(fname, 'r', encoding="utf-8") as ff:
            text = ff.readline()
            
            # タブがあれば消しておきます
            text = text.replace('\t', " ")
            
            text = text+'\t'+'0'+'\t'+'\n'
            f.write(text)

    f.close()
    
    
    # テストデータの作成
    f=open('./data/IMDb_test.tsv','w')

    path = './data/aclImdb/test/pos/'
    for fname in glob.glob(os.path.join(path,'*.txt')):
        with io.open(fname, 'r', encoding="utf-8") as ff:
            text = ff.readline()
            
            # タブがあれば消しておきます
            text = text.replace('\t', " ")
        
            text = text+'\t'+'1'+'\t'+'\n'
            f.write(text)

    path = './data/aclImdb/test/neg/'

    for fname in glob.glob(os.path.join(path,'*.txt')):
        with io.open(fname, 'r', encoding="utf-8") as ff:
            text = ff.readline()
            
            
            # タブがあれば消しておきます
            text = text.replace('\t', " ")
            
            text = text+'\t'+'0'+'\t'+'\n'
            f.write(text)


    f.close()


# 以上
