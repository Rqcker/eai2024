import re
import numpy as np
import pandas as pd
import scipy.sparse as ss
from bs4 import BeautifulSoup
import spacy
import spacy_fastlang
from spacy_cleaner import processing, Cleaner
from corextopic import corextopic as ct
from corextopic import vis_topic as vt
from gensim.corpora.dictionary import Dictionary
from gensim import corpora
from sklearn.feature_extraction.text import CountVectorizer
import optuna
from octis.evaluation_metrics.coherence_metrics import Coherence


def split_documents_by_words(documents, max_words=512):
    """
    Split documents if one document's word count is over than max_words.

    Args:
        documents (list): List of documents as strings.
        max_words (int): Maximum number of words for each document.

    Returns:
        list: List of split documents.
    """
    split_documents = []
    for doc in documents:
        words = doc.split()
        num_words = len(words)
        if num_words <= max_words:
            split_documents.append(doc)
        else:
            # Split document into segments of max_words
            num_segments = num_words // max_words
            for i in range(num_segments + 1):
                start_idx = i * max_words
                end_idx = (i + 1) * max_words
                if ' '.join(words[start_idx:end_idx]) != '' or ' '.join(words[start_idx:end_idx]) != ' ':
                    split_documents.append(' '.join(words[start_idx:end_idx]))
    return split_documents

df = pd.read_csv('/home/yy2046/Workspace/DCEE2023/datasets/twitter/twitter_junhao.csv', encoding='unicode_escape')
data = df['full_text']
    
''' preprocess '''
cleaned_data = []
model = spacy.load("en_core_web_sm")
cleaner = Cleaner( 
    model,
    processing.remove_stopword_token,
    processing.remove_punctuation_token,
    processing.remove_email_token,
    processing.remove_url_token,
    processing.mutate_lemma_token,
    
)

for html_text in data:
    soup = BeautifulSoup(html_text, 'html.parser')
    soup_text = soup.get_text().lower()
    cleaned_data.append(soup_text)
# print(cleaned_data[0])
print('spaCy preprocess start!')
cleaned_data = cleaner.clean(cleaned_data)
print(len(cleaned_data))


docs_list = split_documents_by_words(cleaned_data, max_words=512)

seed_topic_list = [["reduce"], ["reuse"], ["recycle"]]
vectorizer = CountVectorizer(stop_words='english',
                             max_features=20000,
                             binary=True)
search_space = {
    'anchor_strength': [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5],
    'n_hidden': [3, 5, 10, 15, 20, 25]
}

# gridSampler from optuna
sampler = optuna.samplers.GridSampler(search_space)

def objective(trial):
    anchor_strength = trial.suggest_float("anchor_strength", -100, 100)
    n_hidden = trial.suggest_int("n_hidden", -100, 100)

    doc_word = vectorizer.fit_transform(docs_list)
    doc_word = ss.csr_matrix(doc_word)
    words = list(np.asarray(vectorizer.get_feature_names_out()))
    topic_model = ct.Corex(n_hidden=n_hidden,
                        words=words,
                        verbose=False,
                        seed=42)
    topic_model.fit(doc_word,
                    words=words,
                    anchors=seed_topic_list,
                    anchor_strength=anchor_strength)

    '''coherence computation'''
    corpus = [ doc.split(' ') for doc in cleaned_data]
    npmi = Coherence(texts=corpus, topk=10, measure='c_npmi')

    results = topic_model.get_topics()
    # print(results)
    extracted_words = [[item[0] for item in temp] for temp in results]

    # print(extracted_words)

    try:
        npmi_score = npmi.score({'topics':extracted_words})
    except:
        npmi_score = -99

    return npmi_score

study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=len(search_space['anchor_strength']) * len(search_space['n_hidden']))

study.trials_dataframe().to_csv('corex_gs_res_twitts.csv')

# print the best Params
best_trial = study.best_trial
print("Best trial:")
print(f"Value (NPMI): {best_trial.value}")
print("Params: ")
for key, value in best_trial.params.items():
    print(f"{key}: {value}")