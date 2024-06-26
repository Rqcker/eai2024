import re
import pandas as pd
from bs4 import BeautifulSoup
import spacy
import spacy_fastlang
from spacy_cleaner import processing, Cleaner
import gensim
from gensim import corpora
from gensim.models.ldamodel import LdaModel
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

input_data = split_documents_by_words(cleaned_data, max_words=512)
input_tokenized_data = [[token.text for token in model(doc)] for doc in input_data]
input_dictionary = corpora.Dictionary(input_tokenized_data)
input_corpus = [input_dictionary.doc2bow(doc) for doc in input_tokenized_data]


co_tokenized_data = [[token.text for token in model(doc)] for doc in cleaned_data]
co_dictionary = corpora.Dictionary(co_tokenized_data)

n_topics_options = [2, 5, 10, 15, 20, 25]
alpha_options = [0.04, 0.05, 0.07, 0.1, 0.2, 0.5]
eta_options = [0.04, 0.05, 0.07, 0.1, 0.2, 0.5]

# data structure used to store results
results = []

for n_topics in n_topics_options:
    for alpha in alpha_options:
        for eta in eta_options:
            print(f"Training LDA model with n_topics={n_topics}, alpha={alpha}, eta={eta}...")
            lda_model = LdaModel(corpus=input_corpus, id2word=input_dictionary, num_topics=n_topics, 
                                 alpha=alpha, eta=eta, random_state=42, per_word_topics=True)

            try:
                # calculate Coherence score using c_npmi
                coherence_model_lda = CoherenceModel(model=lda_model, texts=co_tokenized_data, 
                                                     dictionary=co_dictionary, coherence='c_npmi')
                coherence_lda = coherence_model_lda.get_coherence()
                print(f"Coherence (c_npmi) score for n_topics={n_topics}, alpha={alpha}, eta={eta}: {coherence_lda}")
            except Exception as e:
                print(f"Failed to calculate coherence for n_topics={n_topics}, alpha={alpha}, eta={eta}. Error: {e}")
                coherence_lda = -99

            # save the results of the current model
            results.append({
                'n_topics': n_topics,
                'alpha': alpha,
                'eta': eta,
                'coherence': coherence_lda
            })

# store as CSV file
results_df = pd.DataFrame(results)
results_df.to_csv('lda_gs_res_twitts.csv', index=False)

print("Optimisation completed. Results are saved to lda_gs_res_twitts.csv")

# print best parameters
best_result = results_df.loc[results_df['coherence'].idxmax()]

print("\nBest Parameters:")
print(f"n_topics: {best_result['n_topics']}, alpha: {best_result['alpha']}, eta: {best_result['eta']}")
print(f"Best Coherence (c_npmi) score: {best_result['coherence']}")
