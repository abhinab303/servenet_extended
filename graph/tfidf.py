import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

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

lemmatizer = WordNetLemmatizer()
english_stopwords = stopwords.words('english')
english_stopwords = set(english_stopwords)


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


categories = [50]
for category in categories:
    # with open(output_folder_path + "/api_dataframe.pickle", "rb") as f:
    #   api_dataframe = pickle.load(f)

    # with open(output_folder_path + "/Common_Files"+"/mashup_dataframe.pickle", "rb") as f:
    #   mashup_dataframe = pickle.load(f)

    api_descriptions = api_dataframe["ServiceDescription"]
    mashup_descriptions = mashup_dataframe["MashupDescription"]
    merged_descriptions = api_descriptions.append(mashup_descriptions)

    cleaned_merged_description_data = []
    for description in merged_descriptions:
        description = description.lower()
        description = description.replace(".", "")
        stopword_removed_description = ' '.join(
            [word for word in description.split(" ") if word not in (english_stopwords)])
        lemmatized_sentence = perform_lemmatization(stopword_removed_description)
        cleaned_merged_description_data.append(lemmatized_sentence)

    tfidf_vectorizer = TfidfVectorizer()
    vectorized_data = tfidf_vectorizer.fit_transform(cleaned_merged_description_data)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    dense_vectorized_data = vectorized_data.todense().A
    # dense_vectorized_data_list = list(dense_vectorized_data)
    # tfidf_dataframe = pd.DataFrame(dense_vectorized_data_list, columns=tfidf_vectorizer.get_feature_names_out())
    tfidf_dataframe = pd.DataFrame(dense_vectorized_data, columns=tfidf_vectorizer.get_feature_names_out())

    with open(output_folder_path + "/tf_idf.pickle", "wb") as f:
        pickle.dump(tfidf_dataframe, f)
