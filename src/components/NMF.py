import os
import sys
import re
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import create_directory
from joblib import dump
from sklearn.decomposition import NMF
from src.utils import topics_document
from collections import Counter
from sklearn.manifold import TSNE
from src.utils import draw_word_cloud, stats_of_documents, show_topic_keywords
from textblob import TextBlob
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook
from bokeh.io import export_png


# Topic Modeling: Non-Negative Matrix Factorization (NMF)
train_documents = pd.read_csv("artifacts/train_set.csv")
test_documents = pd.read_csv("artifacts/test_set.csv")

train_documents['clean_tokens'] = train_documents['clean_document'].progress_apply(lambda x: x.split())
docs = train_documents['clean_tokens']

# NMF is used with Tfidf for optimal results
tfidf_vectorizer = TfidfVectorizer(min_df=3, max_df=0.85,
                                   max_features=4000,
                                   ngram_range=(1, 2), # bi-gram also
                                   preprocessor=" ".join)

vectorized_text = tfidf_vectorizer.fit_transform(docs)

dictionary = Dictionary(docs)

# Filter out words that occur less than 3 documents, or more than 85% of the documents.
# keep_n : Keep only the first 4000 most frequent tokens.
dictionary.filter_extremes(no_below=20, no_above=0.85, keep_n=4000)
# Bag-of-words representation of the documents. (list of (token_id, token_count))
corpus = [dictionary.doc2bow(doc) for doc in docs]

# Create a list of topic-numbers that we want to try
topic_nums = list(np.arange(5,25,))

# Create NMF models for each topic_nums and store their coherence score
coherence_scores = []

for num in topic_nums:
    # NMF model
    # See gensim doc to understand the meaning of these params: https://radimrehurek.com/gensim/models/nmf.html
    nmf = gensim.models.nmf.Nmf(corpus=corpus, num_topics=num,
                                id2word=dictionary, chunksize=2000,
                                passes=5, kappa=0.1, minimum_probability=0.01,
                                w_max_iter=300, w_stop_condition=0.0001,
                                h_max_iter=100, h_stop_condition=0.001,
                                eval_every=10, normalize=True,
                                random_state=42)

    # Coherence Model
    cm = CoherenceModel(model=nmf, texts=docs,
                        dictionary=dictionary, coherence="c_v")

    coherence_scores.append(round(cm.get_coherence(), 5))


# Get the topic number with highest coherence score
scores = list(zip(topic_nums, coherence_scores))

best_topic_number = sorted(scores, key=itemgetter(1), reverse=True)[0][0]

# Plot the results
fig = plt.figure(figsize=(10,6))

plt.plot(topic_nums, coherence_scores,
         linewidth=3, color='#4287f5')

plt.xlabel("Nuber of Topics", fontsize=12)
plt.ylabel("Coherence Score", fontsize=12)
plt.title('Coherence Score by Num. of Topics - Best # Topics: {}'.format(best_topic_number), fontsize=14)

plt.xticks(np.arange(4, max(topic_nums) + 1), fontsize=12)
plt.yticks(fontsize=12)

plt.savefig("artifacts/Figure/Coherence_Score_by_Number")

nmf = NMF(n_components=22, init="nndsvd", random_state=42)
nmf_output = nmf.fit_transform(vectorized_text)

topics = get_topics_terms_weights(nmf_weights, nmf_feature_names)

print_topics_udf(topics=topics, total_topics=2, num_terms=15)
print_topics_udf(topics=topics, total_topics=2, num_terms=10, display_weights=True)

topics_display_list = get_topics_udf(topics, total_topics=2, num_terms=15)
topics_display_list[1]

terms, sizes = getTermsAndSizes(topics_display_list[0])

num_top_words = 15
fontsize_base = 30 / np.max(sizes) # font size for word with largest share in corpus

num_topics = 1

for t in range(num_topics):
    fig, ax = plt.subplots(1, num_topics, figsize=(6, 12))
    plt.ylim(0, num_top_words+0.5)
    plt.axis("off")
    plt.title('Topic #{}'.format(t))

    for i, (word, share) in enumerate(zip(terms, sizes)):
        word = word + " (" + str(share) + ")"
        plt.text(0.03, num_top_words-i, word, fontsize=fontsize_base*share)

plt.tight_layout()

terms, sizes = getTermsAndSizes(topics_display_list[1])

num_top_words = 15
fontsize_base = 50/(np.max(sizes)*0.75) # font size for word with largest share in corpus

num_topics = 1

for t in range(num_topics):
    fig, ax = plt.subplots(1, num_topics, figsize=(8, 15))
    plt.ylim(0, num_top_words+1)
    plt.axis("off")
    plt.title('Topic #{}'.format(t+1))

    for i, (word, share) in enumerate(zip(terms, sizes)):
        word = word + " (" + str(share) + ")"
        plt.text(0.03, num_top_words-i, word, fontsize=fontsize_base*share)

plt.tight_layout();

#Distribution of topics among diff. documents
documet_topic_nmf = topics_document(nmf_output, n_topics=22, data=train_documents)
documet_topic_nmf

nmf_keys = get_keys(nmf_output)
nmf_categories, nmf_counts = keys_to_counts(nmf_keys)

topics_df_nmf = pd.DataFrame({'Topic' : nmf_categories, 'Count' : nmf_counts})
sns.barplot(x=topics_df_nmf['Topic'], y = topics_df_nmf['Count'])
plt.savefig("artifacts/Figure/topics_distribution_nmf.png")

#Top-10 terms in each topic
vocab = tfidf_vectorizer.get_feature_names_out()

for i, comp in enumerate(nmf.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic "+ str(i)+ ": ")
    for t in sorted_words:
        print(t[0], end=", ")
    print("\n")

#Prediction of test documents
test_documents['clean_tokens'] = test_documents['clean_document'].progress_apply(lambda x: x.split())

vectorized_test_text = tfidf_vectorizer.transform(test_documents['clean_tokens'])

nmf_top_test = nmf.transform(vectorized_test_text)

documet_topic_nmf_test = topics_document(nmf_top_test, n_topics=22, data=test_documents)


lsa_model_save_path = "artifacts/models/nmf_model.pkl"
joblib.dump(nmf, lsa_model_save_path)








