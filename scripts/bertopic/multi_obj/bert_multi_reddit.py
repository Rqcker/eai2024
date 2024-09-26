import os
import numpy as np
import pandas as pd
import optuna
import spacy
from bs4 import BeautifulSoup
from optuna.samplers import TPESampler
from spacy_cleaner import processing, Cleaner
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
# from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import InvertedRBO
from umap import UMAP

# Set environment variable to avoid tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def split_documents_by_words(documents, max_words=512):
    """
    Split documents into chunks with a maximum number of words.
    
    Args:
        documents (list): List of documents to be split.
        max_words (int): Maximum number of words per chunk.
        
    Returns:
        list: List of document chunks.
    """
    split_documents = []
    for doc in documents:
        words = doc.split()
        num_words = len(words)
        if num_words <= max_words:
            split_documents.append(doc)
        else:
            num_segments = num_words // max_words
            for i in range(num_segments + 1):
                start_idx = i * max_words
                end_idx = (i + 1) * max_words
                segment = ' '.join(words[start_idx:end_idx])
                if segment.strip():
                    split_documents.append(segment)
    return split_documents 

def calculate_coherence(topic_model, topics, documents):
    """
    Calculate the coherence score using Gensim's CoherenceModel.
    
    Args:
        topic_model (BERTopic): Trained BERTopic model.
        topics (list): List of topics.
        documents (list): List of documents used to fit the model.
        
    Returns:
        float: Coherence score (c_npmi).
    """
    vectoriser = topic_model.vectorizer_model
    analyser = vectoriser.build_analyzer()
    tokens = [analyser(doc) for doc in documents]
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    topic_words = [[words for words, _ in topic_model.get_topic(topic)] for topic in range(len(set(topics)) - 1)]
    
    coherence_model = CoherenceModel(
        topics=topic_words, 
        texts=tokens, 
        corpus=corpus,
        dictionary=dictionary, 
        coherence='c_npmi'
    )
    return coherence_model.get_coherence()

def calculate_diversity(topic_model):
    """
    Calculate the diversity score using InvertedRBO metric.
    
    Args:
        topic_model (BERTopic): Trained BERTopic model.
        
    Returns:
        float: Diversity score.
    """
    topics = topic_model.get_topics()
    topic_words = [[word for word, _ in words] for _, words in topics.items()]
    
    model_output = {"topics": topic_words}
    diversity_model = InvertedRBO()
    
    return diversity_model.score(model_output)

def calculate_perplexity(probs):
    """
    Calculate perplexity based on probability distributions.
    
    Args:
        probs (numpy.array): Array of probabilities from the model.
        
    Returns:
        float: Perplexity score.
    """
    if probs is None or probs.size == 0:
        return float('inf')
    
    probs = np.clip(probs, 1e-10, None)
    log_perplexity = -1 * np.mean(np.log(np.sum(probs, axis=1)))
    
    return np.exp(log_perplexity)

def objective(trial):
    """
    Optuna objective function for hyperparameter optimization of BERTopic model.
    
    Args:
        trial (optuna.Trial): A trial object that contains the hyperparameters.
    
    Returns:
        tuple: c_npmi, diversity, and perplexity scores.
    """
    n_gram = trial.suggest_int("n_gram", 1, 3)
    n_clusters = trial.suggest_int("n_clusters", 2, 25)
    n_components = trial.suggest_int("n_components", 5, 15)
    n_neighbors = trial.suggest_int("n_neighbors", 10, 20)
    
    embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    print(f'Starting trial with n_gram={n_gram}, n_clusters={n_clusters}, n_components={n_components}, n_neighbors={n_neighbors}')
    print('Model supports the max length of a document: ', embedding_model.max_seq_length)
    
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
        calculate_probabilities=True
    )
    
    topics, _ = topic_model.fit_transform(input_data)
    probs, _ = topic_model.approximate_distribution(input_data)

    coherence = calculate_coherence(topic_model, topics, input_data)
    diversity = calculate_diversity(topic_model)
    perplexity = calculate_perplexity(probs)
    
    return coherence, diversity, perplexity

# Load and preprocess data
print('Loading data...')
df = pd.read_json('datasets/reddit/dcee_reddit.json')
df.drop_duplicates(subset=['title', 'selftext'], inplace=True)
data = [row.title + ' ' + str(row.selftext) for index, row in df.iterrows()]

print('Starting preprocessing...')
model = spacy.load("en_core_web_sm")
cleaner = Cleaner(
    model,
    processing.remove_stopword_token,
    processing.remove_punctuation_token,
    processing.remove_email_token,
    processing.remove_url_token,
    processing.mutate_lemma_token,
)

cleaned_data = []
for html_text in data:
    soup = BeautifulSoup(html_text, 'html.parser')
    soup_text = soup.get_text().lower()
    cleaned_data.append(soup_text)

print('spaCy preprocess start!')
cleaned_data = cleaner.clean(cleaned_data)
print('spaCy preprocess done!')

input_data = split_documents_by_words(cleaned_data, max_words=512)
print('Document splitting complete.')

# Perform Optuna hyperparameter optimization
print('Starting hyperparameter optimisation...')
study = optuna.create_study(sampler=TPESampler(), directions=["maximize", "maximize", "minimize"])
study.optimize(objective, n_trials=100)

# Save the optimization results
print('Saving optimization results...')
df_results = study.trials_dataframe()
df_results.to_csv('bertopic_multi_reddit.csv', index=False)
print('Results saved to bertopic_multi_reddit.csv')
