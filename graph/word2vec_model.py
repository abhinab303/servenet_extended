import pandas as pd
from gensim.models import Word2Vec

import nltk
from nltk.corpus import stopwords

import re
import pickle

import json

output_folder_path = "/home/aa7514/PycharmProjects/servenet_extended/files"
ip_file_dir = "/home/aa7514/PycharmProjects/servenet_extended/data/50"
ms_file = "/home/aa7514/PycharmProjects/servenet_extended/data/mashup.txt"
api_file = "/home/aa7514/PycharmProjects/servenet_extended/data/api.txt"

train_df = pd.read_csv(f"{ip_file_dir}/train.csv")
test_df = pd.read_csv(f"{ip_file_dir}/test.csv")
api_dataframe = pd.concat([train_df, test_df], axis=0)

with open(api_file) as f:
    api_json = json.load(f)
api_json = [x for x in api_json if x]

with open(ms_file) as f:
    ms_json = json.load(f)
ms_json = [x for x in ms_json if x]

mashup_dataframe = pd.DataFrame(ms_json)
api_dataframe = pd.DataFrame(api_json)

api_dataframe["ServiceName"] = api_dataframe["ServiceName"].str.replace(' API', '')
mashup_dataframe["MashupName"] = mashup_dataframe["MashupName"].str.replace('Mashup: ', '')

uncleaned_api_dataframe = api_dataframe.copy()
uncleaned_mashup_dataframe = mashup_dataframe.copy()

uncleaned_api_dataframe = uncleaned_api_dataframe[uncleaned_api_dataframe["ServiceName"].notnull()]
uncleaned_api_dataframe = uncleaned_api_dataframe[uncleaned_api_dataframe["ServiceDescription"].notnull()]
uncleaned_api_dataframe = uncleaned_api_dataframe[uncleaned_api_dataframe["ServiceClassification"].notnull()]

uncleaned_mashup_dataframe = uncleaned_mashup_dataframe[uncleaned_mashup_dataframe["MashupName"].notnull()]
uncleaned_mashup_dataframe = uncleaned_mashup_dataframe[uncleaned_mashup_dataframe["MashupDescription"].notnull()]
uncleaned_mashup_dataframe = uncleaned_mashup_dataframe[uncleaned_mashup_dataframe["MashupAPIs"].notnull()]

uncleaned_api_dataframe["ServiceDescription"] = uncleaned_api_dataframe["ServiceDescription"].str.replace(
    r'[^A-Za-z .]+', '', regex=True)
uncleaned_mashup_dataframe["MashupDescription"] = uncleaned_mashup_dataframe["MashupDescription"].str.replace(
    r'[^A-Za-z .]+', '', regex=True)

cleaned_api_dataframe = uncleaned_api_dataframe
cleaned_api_dataframe.index = range(len(cleaned_api_dataframe.index))
cleaned_mashup_dataframe = uncleaned_mashup_dataframe
cleaned_mashup_dataframe.index = range(len(cleaned_mashup_dataframe.index))

english_stopwords = stopwords.words('english')
english_stopwords = set(english_stopwords)

api_description = uncleaned_api_dataframe["ServiceDescription"]
mashup_description = uncleaned_mashup_dataframe["MashupDescription"]
merged_description = api_description.append(mashup_description)

from gensim.models.callbacks import CallbackAny2Vec


class LossLogger(CallbackAny2Vec):
    '''Output loss at each epoch'''

    def __init__(self):
        self.epoch = 1
        self.losses = []

    def on_epoch_begin(self, model):
        print(f'Epoch: {self.epoch}', end='\t')

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        self.losses.append(loss)
        print(f'  Loss: {loss}')
        self.epoch += 1


loss_logger = LossLogger()

sentences = merged_description
sentences = sentences.to_string()
sentences = sentences.lower()
sentences = re.sub('[^a-zA-Z]', ' ', sentences)
sentences = re.sub(r'\s+', ' ', sentences)

all_sentences = nltk.sent_tokenize(sentences)
all_words = [nltk.word_tokenize(sent) for sent in all_sentences]
for i in range(len(all_words)):
    all_words[i] = [w for w in all_words[i] if w not in english_stopwords]

word2vec = Word2Vec(window=5, vector_size=768, min_count=1, epochs=5, workers=4)
word2vec.build_vocab(all_words, progress_per=10000)
word2vec.train(all_words, total_examples=word2vec.corpus_count, epochs=20, compute_loss=True, callbacks=[loss_logger],
               report_delay=1)

print(word2vec.wv.most_similar(positive=['mashup'], topn=5))
print(word2vec.wv.most_similar(positive=['data'], topn=5))

with open(output_folder_path + "/word2vec_model.pickle", "wb") as f:
    pickle.dump(word2vec, f)
