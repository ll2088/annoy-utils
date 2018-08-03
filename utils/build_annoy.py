from annoy import AnnoyIndex
from encoder.use_encoder import USEEncoder
import tensorflow as tf

logging = tf.logging
#FIXME Is this required again?
logging.set_verbosity(logging.INFO)


def build_and_save_annoy(encoder, txt_file, ann_file, num_trees=10, log_freq=100000, batch_size=32):
    if not encoder:
        encoder = USEEncoder()

    ann = AnnoyIndex(encoder.dim())
    ann_index = 0

    sentences = []
    with open(txt_file) as fr:
        for sentence in fr:
            sentence = sentence.strip()
            sentences.append(sentence)

            if len(sentences) == batch_size:
                vectors = encoder.encode(sentences)
                for vector in vectors:
                    ann.add_item(ann_index, vector)
                    ann_index += 1
                sentences = []

            if ann_index % log_freq == 0:
                logging.info(f'{ann_index} Indexed: {ann.get_n_items()}')

    if sentences:
        vectors = encoder.encode(sentences)
        for vector in vectors:
            ann.add_item(ann_index, vector)
            ann_index += 1

    logging.info(f'Final Indexed: {ann.get_n_items()}')
    ann.build(num_trees)
    ann.save(ann_file)
    return ann


def load_annoy(ann_file, dim=512):
    ann = AnnoyIndex(dim)
    ann.load(ann_file)
    logging.info(f'Loading: {ann.get_n_items()}')
    return ann