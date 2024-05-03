import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import create_directory
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from src.utils import topics_document
from collections import Counter
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from wordcloud import WordCloud
from src.utils import draw_word_cloud

# Latent Semantic Analysis(LSA)
train_documents = pd.read_csv("artifacts/train_set.csv")
vectorizer = TfidfVectorizer(stop_words='english',
                             max_features=4000)
vect_text = vectorizer.fit_transform(train_documents['clean_document'])
lsa_model = TruncatedSVD(n_components=10, algorithm='randomized',
                         n_iter=10, random_state=42)
lsa_top = lsa_model.fit_transform(vect_text)

document_topic_lsa = topics_document(model_output=lsa_top,
                                     n_topics=10,
                                     data=train_documents)
plt.bar(x = document_topic_lsa.dominant_topic.value_counts().index,
        height = document_topic_lsa.dominant_topic.value_counts().values)
plt.xlabel('Dominant Topic')
plt.ylabel('Frequency')
plt.title('Distribution of Dominant Topics')
plt.savefig('artifacts/Figure/dominant_topics_bar_plot.png')

vocabulary = vectorizer.get_feature_names_out()
for i, component in enumerate(lsa_model.components_):
    vocab_comp = zip(vocabulary, component)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic_"+ str(i) + ": ")
    for t in sorted_words:
        print(t[0], end=", ")
    print("\n")

# Visualizing topics with t-SNE
# Define helper functions
def get_keys(topic_matrix):
    keys = topic_matrix.argmax(axis=1).tolist()
    return keys

def keys_to_counts(keys):
    count_pairs = Counter(keys).items()
    categories = [pair[0] for pair in count_pairs]
    counts = [pair[1] for pair in count_pairs]
    return (categories, counts)

def get_mean_topic_vectors(keys, tSNE_vectors, n_topics):
    mean_topic_vectors = []
    for t in range(n_topics):
        articles_in_that_topic = []
        for i in range(len(keys)):
            if keys[i] == t:
                articles_in_that_topic.append(tSNE_vectors[i])
        articles_in_that_topic = np.vstack(articles_in_that_topic)
        mean_article_in_that_topic = np.mean(articles_in_that_topic, axis=0)
        mean_topic_vectors.append(mean_article_in_that_topic)
    return mean_topic_vectors
colormap = np.array(["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
                     "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
                     "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
                     "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5" ])

colormap = colormap[:10]
lsa_keys = get_keys(lsa_top)
lsa_categories, lsa_counts = keys_to_counts(lsa_keys)

tsne_lsa_model = TSNE(n_components=2, perplexity=50,
                      learning_rate=100, n_iter=2000, angle=0.75,
                      verbose=0, random_state=0)

tsne_lsa_vectors = tsne_lsa_model.fit_transform(lsa_top)
lsa_mean_topic_vectors = get_mean_topic_vectors(lsa_keys, tsne_lsa_vectors, n_topics=10)

plot = figure(title="t-SNE Clustering of {} LSA Topics".format(10),
              width=700, height=700)  # Use width and height instead of outer_width and outer_height

plot.scatter(x=tsne_lsa_vectors[:,0],
             y=tsne_lsa_vectors[:,1],
             color=colormap[lsa_keys])

for t in range(10):
    label = Label(x=lsa_mean_topic_vectors[t][0],
                  y=lsa_mean_topic_vectors[t][1],
                  text_color=colormap[t])

    plot.add_layout(label)

output_file_path = "artifacts/Figure/tsne_clustering_LSA_topics.png"
export_png(plot, filename=output_file_path)

# Generating Word-Clouds
vocabulary = vectorizer.get_feature_names_out()
# Word Cloud for topic-0
draw_word_cloud(topic_index=0, model=lsa_model)
draw_word_cloud(1, lsa_model)
draw_word_cloud(5, lsa_model)




