import numpy as np
from w2v_utils import *

words, word_to_vec_map = read_glove_vecs('./data/glove.6B.50d.txt')


def cosine_similarity(A, B):
    """
    Cosine similarity reflects the degree of similarity between u and v

    Arguments:
        A -- a word vector of shape (n,)
        B -- a word vector of shape (n,)

    Returns:
        cosine_similarity -- the cosine similarity between u and v defined by the formula above.
    """

    dot = np.dot(A, B)
    norma = np.linalg.norm(A)
    normb = np.linalg.norm(B)
    cos = dot/(norma*normb)
    return cos

def analogy(word_a, word_b, word_c, word_to_vec_map):
    """
    Performs the word analogy task as explained above: a is to b as c is to ____.

    Arguments:
    word_a -- a word, string
    word_b -- a word, string
    word_c -- a word, string
    word_to_vec_map -- dictionary that maps words to their corresponding vectors.

    Returns:
    best_word --  the word such that v_b - v_a is close to v_best_word - v_c, as measured by cosine similarity
    """
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()
    e_a, e_b, e_c = word_to_vec_map[word_a], word_to_vec_map[word_b], word_to_vec_map[word_c]
    words = word_to_vec_map.keys()
    max_cosine_sim = -100              # Initialize max_cosine_sim to a large negative number
    best_word = None                   # Initialize best_word with None, it will help keep track of the word to output
    input_words_set = set([word_a, word_b, word_c])
    for w in words:
        if w in input_words_set:
            continue
        # Compute cosine similarity between the vector (e_b - e_a) and the vector ((w's vector representation) - e_c)
        cosine_sim = cosine_similarity(word_to_vec_map[w]-e_c, e_b-e_a)
        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = w
    return best_word

print('words length: {}'.format(len(words)))
print('word_to_vec_map[the]: {}'.format(np.array(word_to_vec_map['word']).shape))


triads_to_try = [('man', 'strength', 'woman'),
                 ('Egypt', 'History', 'Germany'),
                 ('Egypt', 'Nile', 'Iraq') ]


for triad in triads_to_try:
    print ('{} -> {} :: {} -> {}'.format( *triad, analogy(*triad,word_to_vec_map)))
