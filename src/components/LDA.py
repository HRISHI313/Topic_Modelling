import os
import sys
import re
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import create_directory
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import TruncatedSVD
from src.utils import topics_document
from collections import Counter
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from wordcloud import WordCloud
from src.utils import draw_word_cloud, stats_of_documents, show_topic_keywords
from textblob import TextBlob
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook
from bokeh.io import export_png
import gensim
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel



# Topic Modelling: Latent Dirichlet Allocation(LDA)
train_documents = pd.read_csv("artifacts/train_set.csv")
test_documents = pd.read_csv("artifacts/test_set.csv")

from sklearn.feature_extraction.text import CountVectorizer

# Using count vectorizer to build the LDA model
vectorizer = CountVectorizer(analyzer="word",
                             min_df = 10,
                             stop_words="english",
                             lowercase=True)
vectorized_data = vectorizer.fit_transform(train_documents['clean_document'])
lda_model = LatentDirichletAllocation(n_components=20, # Num of topics
                                      max_iter=10, # Max learning iterations
                                      learning_method="online",
                                      batch_size=128,
                                      evaluate_every=-1, # See doc, How often to evaluate perplexity
                                      random_state=42,
                                      n_jobs=-1)

lda_output = lda_model.fit_transform(vectorized_data)

# Log likelihood: Higher the better
print(f"Log Likelihood: {lda_model.score(vectorized_data)}")
# Perplexity: Lower the better
print(f"Perplexity: {lda_model.perplexity(vectorized_data)}")

# Grid Search
params = {'n_components': [10,15,20],
          'learning_decay': [.5, .7, .9]}
# Instantiate the model
lda = LatentDirichletAllocation(max_iter=5,
                                learning_method="online",
                                learning_offset=50.,
                                random_state=42)
lda_gs = GridSearchCV(estimator=lda, param_grid=params, n_jobs=-1)
lda_gs.fit(vectorized_data)

# Best Model
best_lda_model = lda_gs.best_estimator_
# Model Parameters
print("Best Model's Params: ", lda_gs.best_params_)
# Log Likelihood Score
print("Best Log Likelihood Score: ", lda_gs.best_score_)
# Perplexity
print("Model Perplexity: ", best_lda_model.perplexity(vectorized_data))

lda_output = best_lda_model.transform(vectorized_data)

document_topic_lda = topics_document(model_output=lda_output, n_topics=10, data=train_documents)
document_topic_lda

document_topic_lda.dominant_topic.value_counts()

lda_keys = get_keys(lda_output)
# print(lda_keys[:10])
lda_categories, lda_counts = keys_to_counts(lda_keys)
# print(lda_categories, lda_counts)
topics_df_lda = pd.DataFrame({'topic' : lda_categories, 'count' : lda_counts})
sns.barplot(x=topics_df_lda['topic'], y = topics_df_lda['count'])

save_path = "artifacts/Figure/topics_distribution_lda.png"
plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
print(f"Bar plot saved as {save_path}")

topic_keywords = show_topic_keywords(vectorizer = vectorizer,
                                     model = best_lda_model,
                                     top_n_words=15)
topic_keywords_df = pd.DataFrame(topic_keywords)

topic_keywords_df.columns = ['Word '+ str(i) for i in range(topic_keywords_df.shape[1])]
topic_keywords_df.index = ['Topic '+ str(i) for i in range(topic_keywords_df.shape[0])]

tsne_lda_model = TSNE(n_components=2, perplexity=50,
                      learning_rate=100, n_iter=2000, angle=0.75,
                      verbose=0, random_state=0)

tsne_lda_vectors = tsne_lda_model.fit_transform(lda_output)

colormap = np.array(["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
                     "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
                     "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
                     "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5" ])

colormap = colormap[:10]

lda_mean_topic_vectors = get_mean_topic_vectors(lda_keys, tsne_lda_vectors, n_topics=10)

plot = figure(title="t-SNE Clustering of {} LDA Topics".format(10),
              width=700, height=700)  # Use width and height instead of outer_width and outer_height

plot.scatter(x=tsne_lda_vectors[:,0],
             y=tsne_lda_vectors[:,1],
             color=colormap[lda_keys])

for t in range(10):
    label = Label(x=lda_mean_topic_vectors[t][0],
                  y=lda_mean_topic_vectors[t][1],
                  text_color=colormap[t])

    plot.add_layout(label)

output_png_path = "artifacts/Figure/tSNE_clustering_LDA_topics.png"
export_png(plot, filename=output_png_path)

# LDA - gensim
# Simply converting cleaaaned text into list of words/tokens
train_documents['clean_tokens'] = train_documents['clean_document'].progress_apply(lambda x: x.split())

# Building LDA Model
lda_model_gensim = gensim.models.LdaMulticore(corpus=corpus,
                                              num_topics=10,
                                              id2word=dictionary,
                                              chunksize=1000,
                                              passes=10,
                                              iterations=100,
                                              per_word_topics=True,
                                              random_state=42,
                                              workers=None)

# Compute Coherennce score
coherence_model_lda = CoherenceModel(model=lda_model_gensim,
                                     texts=docs,
                                     dictionary=dictionary,
                                     coherence="c_v")

coherence_lda = coherence_model_lda.get_coherence()
print('Coherence Score: %.4f' % coherence_lda)

topic_keywords_gensim = []

for i in range(10):
    topic_i = lda_model_gensim.print_topic(topicno=i, topn=15)
    topic_i = re.findall(r'"([^"]+)"', topic_i)
    topic_keywords_gensim.append(topic_i)

# topic-keyword dataframe
topic_keywords_gensim_df = pd.DataFrame(topic_keywords_gensim)

topic_keywords_gensim_df.columns = ['Word '+ str(i) for i in range(topic_keywords_gensim_df.shape[1])]
topic_keywords_gensim_df.index = ['Topic '+ str(i) for i in range(topic_keywords_gensim_df.shape[0])]

lda_model_gensim.save("artifacts/lda_model.pkl")













