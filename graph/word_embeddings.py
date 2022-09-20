import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from gensim import models

import pickle

import json

output_folder_path = "/home/aa7514/PycharmProjects/servenet_extended/files"
ip_file_dir = "/home/aa7514/PycharmProjects/servenet_extended/data/50"
ms_file = "/home/aa7514/PycharmProjects/servenet_extended/data/mashup.txt"

train_df = pd.read_csv(f"{ip_file_dir}/train.csv")
test_df = pd.read_csv(f"{ip_file_dir}/test.csv")
api_dataframe = pd.concat([train_df, test_df], axis=0)

with open(ms_file) as f:
    ms_json = json.load(f)
ms_json = [x for x in ms_json if x]

mashup_dataframe = pd.DataFrame(ms_json)

api_dataframe["ServiceName"] = api_dataframe["ServiceName"].str.replace(' API', '')
mashup_dataframe["MashupName"] = mashup_dataframe["MashupName"].str.replace('Mashup: ', '')
api_dataframe["ServiceDescription"] = api_dataframe["ServiceDescription"].str.replace(r'[^A-Za-z .]+', '', regex=True)
mashup_dataframe["MashupDescription"] = mashup_dataframe["MashupDescription"].str.replace(r'[^A-Za-z .]+', '',
                                                                                          regex=True)
api_dataframe.reset_index(inplace=True)
mashup_dataframe.reset_index(inplace=True)


def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


lemmatizer = WordNetLemmatizer()
english_stopwords = stopwords.words('english')
english_stopwords = set(english_stopwords)


def compute_unique_word_dictionary():
    lemmatized_dictionary = {}
    unique_words_dictionary = {}
    for description in merged_descriptions:
        pos_tagged = nltk.pos_tag(nltk.word_tokenize(description))
        wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
        for word, tag in wordnet_tagged:
            word = word.lower()
            if word in english_stopwords or word == "" or word == "." or word == "api":
                continue
            if tag is None:
                lemmatized_word = lemmatizer.lemmatize(word)
            else:
                lemmatized_word = lemmatizer.lemmatize(word, tag)

            if lemmatized_word not in lemmatized_dictionary.keys():
                lemmatized_dictionary[lemmatized_word] = set()
            lemmatized_dictionary[lemmatized_word].add(word)

            unique_words_dictionary[lemmatized_word] = unique_words_dictionary.get(lemmatized_word, 0) + 1

    filtered_unique_words = {}
    for key, value in unique_words_dictionary.items():
        if value < 5 or len(key) < 2:
            continue
        filtered_unique_words[key] = value
    return filtered_unique_words, lemmatized_dictionary


def perform_lemmatization(sentence):
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_sentence.append(word)
        else:
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    lemmatized_sentence = " ".join(lemmatized_sentence)
    return lemmatized_sentence

# with open(output_folder_path + "/word2vec_model.pickle", "rb") as f:
#   word2vec = pickle.load(f)

word2vec = models.KeyedVectors.load_word2vec_format(
    "/home/aa7514/PycharmProjects/servenet_extended/files/GoogleNews-vectors-negative300.bin.gz",
    binary=True)

api_descriptions = api_dataframe["ServiceDescription"]
mashup_descriptions = mashup_dataframe["MashupDescription"]
merged_descriptions = api_descriptions.append(mashup_descriptions)

unique_words_in_corpus, lemmatized_dictionary = compute_unique_word_dictionary()

word_feature_vectors = []
for word in unique_words_in_corpus.keys():
    is_present = False
    for non_lemma_word in lemmatized_dictionary[word]:
        try:
            # vector = word2vec.wv[non_lemma_word]
            vector = word2vec[non_lemma_word]
            word_feature_vectors.append(vector)
            is_present = True
            break
        except:
            continue
    if not is_present:
        vector = [0] * 768
        word_feature_vectors.append(vector)

with open(output_folder_path + "/unique_words_in_corpus.pickle", "wb") as f:
    pickle.dump(unique_words_in_corpus, f)
with open(output_folder_path + "/lemmatized_dictionary.pickle", "wb") as f:
    pickle.dump(lemmatized_dictionary, f)
with open(output_folder_path + "/word_embeddings.pickle", "wb") as f:
    pickle.dump(word_feature_vectors, f)
