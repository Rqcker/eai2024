import pandas as pd
from bs4 import BeautifulSoup
import optuna
import spacy
from spacy_cleaner import processing, Cleaner
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired
from octis.evaluation_metrics.coherence_metrics import Coherence
from umap import UMAP
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel


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
 
'''bertopic model and hpo'''
def objective(trial):
    
    n_gram = trial.suggest_float("n_gram", -100, 100)
    n_clusters = trial.suggest_int("n_clusters", -100, 100)
    n_components = trial.suggest_int("n_components", -100, 100)
    n_neighbors = trial.suggest_int("n_neighbors", -100, 100)
    
    embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    print('Model supported the max length of a document: ', embedding_model.max_seq_length)
    umap_model = UMAP(n_neighbors=n_neighbors, n_components=n_components, random_state=42)
    cluster_model = KMeans(n_clusters=n_clusters, random_state=42)
    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, n_gram))
    representation_model = KeyBERTInspired(top_n_words=10, random_state=42)
    
    topic_model = BERTopic(
        embedding_model=embedding_model,
        top_n_words=10,
        umap_model=umap_model,
        hdbscan_model=cluster_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,

    )
    topics, probs = topic_model.fit_transform(input_data)
    print(topic_model.get_topic_info())

    '''coherence computation'''
    # corpus = [ doc.split(' ') for doc in cleaned_data]
    # npmi = Coherence(texts=corpus, topk=10, measure='c_npmi')

    # results = topic_model.get_topics()
    # extracted_words = []
    # for key, word_list in results.items():
    #     tmp = []
    #     for word, _ in word_list:
    #         tmp.append(word)
    #     extracted_words.append(tmp)
    # # print(extracted_words)
    vectorizer = topic_model.vectorizer_model
    analyzer = vectorizer.build_analyzer()
    words = vectorizer.get_feature_names()
    tokens = [analyzer(doc) for doc in cleaned_data]
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    topic_words = [[words for words, _ in topic_model.get_topic(topic)] 
               for topic in range(len(set(topics))-1)]
    coherence_model = CoherenceModel(topics=topic_words, 
                                 texts=tokens, 
                                 corpus=corpus,
                                 dictionary=dictionary, 
                                 coherence='c_npmi')

    coherence = coherence_model.get_coherence()

    # try:
    #     npmi_score = npmi.score({'topics':extracted_words})
    # except ValueError:
    #     npmi_score = None
    # # print("Coherence: "+str(npmi_score))
    return coherence


df = pd.read_json('/home/yy2046/Workspace/DCEE2023/datasets/reddit/subreddit_posts_updated.json')
df.drop_duplicates(subset=['title', 'selftext'], inplace=True)
data = [row.title + ' ' + str(row.selftext) for index, row in df.iterrows()]

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
# print(cleaned_data[0])
print('spaCy preprocess done!')

# max_seq_length = 0
# for doc in cleaned_data:
#     if len(doc.split(' ')) > max_seq_length:
#         max_seq_length = len(doc.split(' '))
# print(max_seq_length)
 
 
input_data = split_documents_by_words(cleaned_data, max_words=512)

search_space = {"n_gram": [1, 2, 3], 'n_clusters': [2, 5, 10, 15, 20, 25], 'n_components': [5, 10, 15], 'n_neighbors': [10, 15, 20]}
study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space), direction="maximize")
study.optimize(objective)
df = study.trials_dataframe()
df.to_csv('bertopic_gs_res_reddits.csv', index=False)
