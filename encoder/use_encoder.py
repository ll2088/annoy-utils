import tensorflow as tf
import tensorflow_hub as hub

logging = tf.logging
logging.set_verbosity(logging.ERROR)


TFHUB_URL_USE = 'https://tfhub.dev/google/universal-sentence-encoder/1'
TFHUB_URL_USE_LARGE = 'https://tfhub.dev/google/universal-sentence-encoder-large/1'


class USEEncoder(object):
    def __init__(self, large=False):
        if large:
            url = TFHUB_URL_USE_LARGE
        else:
            url = TFHUB_URL_USE
        embed = hub.Module(url)
        self.sentences = tf.placeholder(dtype=tf.string, shape=[None])

        self.embedding_fun = embed(self.sentences)
        self.sess = tf.Session()
        self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        print(f'USE Encoder Ready! large: {large}')

    def encode(self, sentences):
        if not sentences:
            return None
        return self.sess.run(self.embedding_fun, feed_dict={self.sentences: sentences})