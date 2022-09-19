import pandas as pd
import json

import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

import re
import pickle
import networkx as nx

lemmatizer = WordNetLemmatizer()
english_stopwords = stopwords.words('english')
english_stopwords = set(english_stopwords)

category_num = 50
ip_file_dir = "/home/aa7514/PycharmProjects/servenet_extended/data/"
train_file = f"{ip_file_dir}{category_num}/train.csv"
test_file = f"{ip_file_dir}{category_num}/test.csv"
mashup_file = f"{ip_file_dir}mashup.txt"

file_path = "/home/aa7514/PycharmProjects/servenet_extended/files"


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


def load_data():
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    api_df = pd.concat([train_df, test_df], axis=0)

    with open(mashup_file) as f:
        ms_json = json.load(f)
    ms_json = [x for x in ms_json if x]

    mashup_df = pd.DataFrame(ms_json)

    api_df["ServiceName"] = api_df["ServiceName"].str.replace(' API', '')
    mashup_df["MashupName"] = mashup_df["MashupName"].str.replace('Mashup: ', '')
    api_df["ServiceDescription"] = api_df["ServiceDescription"].str.replace(r'[^A-Za-z .]+', '', regex=True)
    mashup_df["MashupDescription"] = mashup_df["MashupDescription"].str.replace(r'[^A-Za-z .]+', '', regex=True)
    api_df.reset_index(inplace=True, drop=True)
    mashup_df.reset_index(inplace=True, drop=True)

    api_desc = api_df["ServiceDescription"]
    mashup_desc = mashup_df["MashupDescription"]
    merged_desc = api_desc.append(mashup_desc)
    merged_desc.reset_index(drop=True, inplace=True)

    return api_df, mashup_df, merged_desc


def construct_graph_nodes(graph, cleaned_api_dataframe, cleaned_mashup_dataframe, unique_words_in_corpus):
    # # Iterate over API dataframe. Add node for each record based on index
    for index, row in cleaned_api_dataframe.iterrows():
        graph.add_node(index)
        api_name = row[0].strip().lower()
        node_to_api[index] = api_name
        api_to_node[api_name] = index

    # # Continue iteration
    checkpointIndex = len(cleaned_api_dataframe)

    # # Iterate over Mashup dataframe and add node for each record based on index
    for index, row in cleaned_mashup_dataframe.iterrows():
        graph.add_node(checkpointIndex + index)
        mashup_name = row[0].strip().lower()
        node_to_api[checkpointIndex + index] = mashup_name
        api_to_node[mashup_name] = checkpointIndex + index

    # # continue iteration
    checkpointIndex = len(cleaned_api_dataframe) + len(cleaned_mashup_dataframe)

    # # Iterate over unique words dictionary and add word nodes
    for word in unique_words_in_corpus:
        graph.add_node(checkpointIndex)
        node_to_word[checkpointIndex] = word
        word_to_node[word] = checkpointIndex
        checkpointIndex = checkpointIndex + 1


def construct_mashup_api_edges(graph, cleaned_api_dataframe, cleaned_mashup_dataframe, api_to_node):
    count = 0
    checkpointIndex = len(cleaned_api_dataframe)
    for index, row in cleaned_mashup_dataframe.iterrows():
        graph.add_edge(checkpointIndex + index, checkpointIndex + index)
        api_called_data = row[2]

        # # For all the api invoked by mashup, create edge between mashup and that particular api
        for api_data in api_called_data:
            api_data = str(api_data)
            api_name = api_data.strip().lower()
            if api_name in api_to_node:
                print(".", end="")
                count = count + 2
                api_index_id = api_to_node[api_name]
                weight_value = 1 - (1 / len(api_called_data))
                graph.add_edge(checkpointIndex + index, api_index_id, weight=weight_value)
                graph.add_edge(api_index_id, checkpointIndex + index, weight=weight_value)


def construct_api_word_edges(graph, cleaned_api_dataframe, unique_words_in_corpus, tfidf_dataframe):
    count = 0
    for index, row in cleaned_api_dataframe.iterrows():
        graph.add_edge(index, index)

        # obtain the api description
        description = row[1]

        # remove any special characters from the description
        description = re.sub(r'[^A-Za-z0-9 ]+', '', description)

        # create a set of unique words from the description
        words = set(description.strip().split(" "))

        for word in words:
            if word == "":
                continue
            word = word.lower()

            # # Associate the Tag
            pos_tagged = nltk.pos_tag(nltk.word_tokenize(word))

            # # Obtain the word and tag
            wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))

            current_word, current_word_tag = wordnet_tagged[0][0], wordnet_tagged[0][1]

            # # Do not process a word if stopword or empty or "." or "api" or not in vocabulary
            if current_word in english_stopwords or current_word == "" or current_word == "." or current_word == "api" or current_word not in unique_words_in_corpus.keys():
                continue
            if current_word_tag is None:
                lemmatized_word = lemmatizer.lemmatize(current_word)
            else:
                lemmatized_word = lemmatizer.lemmatize(current_word, current_word_tag)

            # # If word in dictionary
            # if current_word in word_to_node:
            if lemmatized_word in word_to_node:
                word_node_index = word_to_node[current_word]
                try:
                    graph.add_edge(index, word_node_index, weight=tfidf_dataframe.loc[index, word])
                    graph.add_edge(word_node_index, index, weight=tfidf_dataframe.loc[index, word])
                    count = count + 2
                    print("a", end="")
                except Exception as e:
                    count = count + 1
                    graph.add_edge(index, word_node_index, weight=0.25)


def construct_mashup_word_edges(graph, cleaned_api_dataframe, cleaned_mashup_dataframe, unique_words_in_corpus,
                                tfidf_dataframe):
    count = 0
    checkpointIndex = len(cleaned_api_dataframe)
    for index, row in cleaned_mashup_dataframe.iterrows():
        graph.add_edge(checkpointIndex + index, checkpointIndex + index)
        description = row[1]
        description = re.sub(r'[^A-Za-z0-9 ]+', '', description)
        words = set(description.strip().split(" "))
        for word in words:
            if word == "":
                continue
            word = word.lower()
            pos_tagged = nltk.pos_tag(nltk.word_tokenize(word))
            wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
            current_word, current_word_tag = wordnet_tagged[0][0], wordnet_tagged[0][1]
            if current_word in english_stopwords or current_word == "" or current_word == "." or current_word == "api" or current_word not in unique_words_in_corpus.keys():
                continue
            if current_word_tag is None:
                lemmatized_word = lemmatizer.lemmatize(current_word)
            else:
                lemmatized_word = lemmatizer.lemmatize(current_word, current_word_tag)

            # if current_word in word_to_node:
            if lemmatized_word in word_to_node:
                word_node_index = word_to_node[current_word]
                try:
                    graph.add_edge(checkpointIndex + index, word_node_index,
                                   weight=tfidf_dataframe.loc[checkpointIndex + index, word])
                    graph.add_edge(word_node_index, checkpointIndex + index,
                                   weight=tfidf_dataframe.loc[checkpointIndex + index, word])
                    count = count + 2
                    print("m", end="")
                except Exception as e:
                    count = count + 1
                    graph.add_edge(checkpointIndex + index, word_node_index, weight=0.25)


def construct_word_word_edges(graph, cleaned_api_dataframe, cleaned_mashup_dataframe, unique_words_in_corpus,
                              lemmatized_dictionary, word2vec):
    count = 0
    checkpointIndex = len(cleaned_api_dataframe) + len(cleaned_mashup_dataframe)
    word_list = list(unique_words_in_corpus.keys())

    # # Iterate over each word in the unique words dictionary
    for word in word_list:
        if word == "":
            continue
        graph.add_edge(checkpointIndex, checkpointIndex)

        # # For all possible word combination for a given root word
        for non_lemma_word in lemmatized_dictionary[word]:
            is_present = False
            try:
                word_1_index = word_to_node[non_lemma_word]

                # # Find similar word using word2vec model
                for similar_word in word2vec.wv.most_similar(positive=[word], topn=5):
                    pos_tagged = nltk.pos_tag(nltk.word_tokenize(similar_word[0]))
                    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
                    current_word, current_word_tag = wordnet_tagged[0][0], wordnet_tagged[0][1]

                    # # Do not process if it is a stop word or empty string or symbol or word api or not in unique words dictionary
                    if current_word in english_stopwords or current_word == "" or current_word == "." or current_word == "api" or current_word not in unique_words_in_corpus.keys():
                        continue

                        # # If tag is none, process without tag, else process with tag info
                    if current_word_tag is None:
                        lemmatized_word = lemmatizer.lemmatize(current_word)
                    else:
                        lemmatized_word = lemmatizer.lemmatize(current_word, current_word_tag)

                    # # Add edge
                    try:
                        word_2_index = word_to_node[lemmatized_word]
                        similarity_score = similar_word[1]
                        if similarity_score > 0.5:
                            print("w", end="")
                            count = count + 1
                            graph.add_edge(word_1_index, word_2_index, weight=similarity_score)
                            is_present = True
                            break
                    except Exception as e:
                        continue

            except Exception as e:
                continue
            if is_present:
                break


node_to_api = {}
api_to_node = {}
node_to_word = {}
word_to_node = {}

if __name__ == "__main__":
    api_dataframe, mashup_dataframe, merged_descriptions = load_data()

    with open(file_path + "/tf_idf.pickle", "rb") as f:
        tfidf_dataframe = pickle.load(f)

        #   with open(output_folder_path + "/sentence_embeddings.pickle", "rb") as f:
        #     sentence_embeddings = pickle.load(f)
    # sentence_embeddings = sentence_vectors

    with open(file_path + "/unique_words_in_corpus.pickle", "rb") as f:
        unique_words_in_corpus = pickle.load(f)

    with open(file_path + "/lemmatized_dictionary.pickle", "rb") as f:
        lemmatized_dictionary = pickle.load(f)

    with open(file_path + "/word2vec_model.pickle", "rb") as f:
        word2vec = pickle.load(f)

    with open(file_path + "/word_embeddings.pickle", "rb") as f:
        word_embeddings = pickle.load(f)

    graph = nx.Graph()

    print("construct_graph_nodes")
    construct_graph_nodes(graph, api_dataframe, mashup_dataframe, unique_words_in_corpus)
    print("construct_mashup_api_edges")
    construct_mashup_api_edges(graph, api_dataframe, mashup_dataframe, unique_words_in_corpus)
    print("construct_api_word_edges")
    # construct_api_word_edges(graph, api_dataframe, unique_words_in_corpus, tfidf_dataframe)
    print("construct_mashup_word_edges")
    construct_mashup_word_edges(graph, api_dataframe, mashup_dataframe, unique_words_in_corpus, tfidf_dataframe)
    print("construct_word_word_edges")
    construct_word_word_edges(graph, api_dataframe, mashup_dataframe, unique_words_in_corpus, lemmatized_dictionary,
                              word2vec)

    # feature_matrix = np.append(sentence_embeddings, word_embeddings, axis=0)

    with open(file_path + "/graph.pickle", "wb") as f:
        pickle.dump(graph, f)

    # with open(file_path + "/feature_matrix.pickle", "wb") as f:
    #     pickle.dump(word_embeddings, f)
