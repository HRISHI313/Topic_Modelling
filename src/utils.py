import os
import sys
from src.logger import logging
from src.exception import CustomException




# CREATING DIRECTORIES:
def create_directory(path:str):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
            logging.info(f"Directory [{path}] has been created")
        else:
            logging.info(f"Directory [{path}] already exists")
    except CustomException as e:
        logging.error(f"Error in create_directory: {e}")


def topics_document(model_output, n_topics , data):
    topicnames = ["Topic_" + str(i) for i in range(n_topics)]
    docnames = ["Doc_" + str(i) for i in range(len(data))]
    df_document_topic = pd.DataFrame(np.round(model_output, 2), columns=topicnames, index=docnames)
    dominant_topic = np.argmax(df_document_topic.values, axis=1)
    df_document_topic["dominant_topic"] = dominant_topic
    return df_document_topic

def draw_word_cloud(topic_index, model):
    imp_words_topic=""
    component = model.components_[topic_index]
    vocab_comp = zip(vocabulary, component)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:50]

    for word in sorted_words:
        imp_words_topic=imp_words_topic+" "+word[0]
    wordcloud = WordCloud(width=600, height=400).generate(imp_words_topic)
    plt.figure( figsize=(5,5))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def stats_of_documents(data:pd.DataFrame):

    data = data['clean_document'].tolist()

    # PoS tagging
    tagged_documents = [TextBlob(data[i]).pos_tags for i in range(len(data))]
    tagged_documents_df = pd.DataFrame({'tags': tagged_documents})

    word_counts = []
    pos_counts = {}

    for tagged_text in tagged_documents_df['tags']:
        word_counts.append(len(tagged_text))

        for word, tag in tagged_text:
            if tag in pos_counts:
                pos_counts[tag] += 1
            else:
                pos_counts[tag] = 1

    print(f"Total number of words in the document: {np.sum(word_counts)}")
    print(f"Mean count of words per article in the document: {round(np.mean(word_counts))}")
    print(f"Minimum count of words of an article in the document: {np.min(word_counts)}")
    print(f"Maxmimum count of words of an article in the document: {np.max(word_counts)}")

    pos_sorted_types = sorted(pos_counts, key=pos_counts.__getitem__, reverse=True)
    pos_sorted_counts = sorted(pos_counts.values(), reverse=True)

    fig, ax = plt.subplots(figsize=(12,4))
    ax.bar(range(len(pos_counts)), pos_sorted_counts)
    ax.set_xticks(range(len(pos_counts)))
    ax.set_xticklabels(pos_sorted_types, rotation=30)
    ax.set_title('Part-of-Speech Tagging for the Corpus of Articles')
    ax.set_xlabel('Type of Word')
    plt.show();