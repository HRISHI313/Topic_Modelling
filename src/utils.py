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