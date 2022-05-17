# naive_bayes.py
import argparse
from collections import Counter
import os
from pathlib import Path
import pandas as pd
import pickle
import re
import math
import nltk
import numpy as np
import string

from src.nb_args import NaiveBayesArgs

DATA_DIR = '.data'


def get_arguments():
    parser = argparse.ArgumentParser(description='Multinomial Naїve Bayes Classifier')
    parser.add_argument('-s', '--src_path', type=str, default=DATA_DIR,
                        help='Source path for processing.')
    parser.add_argument('-od', '--output_dir', type=str, default=Path(__file__).parent,
                        help='Working directory for saving files etc')
    parser.add_argument('-td', '--train_data', type=str, help='Training data')
    parser.add_argument('-vd', '--val_data', type=str, help='Validation data')
    parser.add_argument('-c', '--classes', type=str, default='tpole_labels.txt',
                        help='All classes')
    parser.add_argument('-sl', '--save_label', type=str, default='',
                        help='Optional label to add to save name for easier identification.')
    parser.add_argument('-sm', '--to_save_model', action='store_true',
                        help='Flag for whether to save the model vocabulary.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Flag for whether to print lots of text data.')

    args = parser.parse_args()

    return NaiveBayesArgs(args)


def log_p(n):
    # Get the probability's natural log of n, return as -99 if n == 0
    # This is accurate enough for our purposes and avoids blowing up when n==0. EXP(-99) is about 10e-43
    return math.log(n) if n != 0 else LOG_ZERO


# def get_entropy(p_arr):
#     # Entropy H(X) = -SUM (p log p)
#     # The contribution of each class is -p log p
#     individual = np.empty(0)
#     for p in p_arr:
#         individual = np.append(individual, -p_log_p(p))
#
#     overall = np.sum(individual)
#
#     return overall, individual


def get_class_documents(documents, doc_labels, k):
    # Get documents belonging to the k-th class
    class_docs = list()
    for i in range(len(documents)):
        if k == doc_labels[i]:
            class_docs.append(documents[i])

    return class_docs


def get_count_of_term_in_docs(documents, w):
    # Get the count of the term in the documents
    count_wt = np.array([len([term for term in t[1] if w == term]) for t in documents])

    return len(np.array([word_freq for word_freq in count_wt if word_freq > 0]))


def get_laplace_prob_term(class_vocabulary, w, m):
    """
    Return Laplacian probability of the term w appearing in class_vocabulary.

    Laplacian p(wi|ck) = ((wi for ck) + 1) / (N + K)

    where N is count of all words in the class and K is the number of classes.

    :parameter class_vocabulary: body of tokenized documents
    :parameter w: the subject term or word
    :parameter m: number of classes
    """
    w_count = [v[2] for v in class_vocabulary if v[1] == w]
    w_count = 0 if len(w_count) == 0 else w_count[0]
    v_sum = np.sum([v[2] for v in class_vocabulary])
    return (w_count + 1) * 1.0 / (v_sum + m)


def build_model(documents, labels, to_stem):
    # Build the class prediction model, based on the corpus of documents

    # Tokenize
    documents = np.array([np.array([labels[i], tokenize(documents[i], to_stem)],
                                   dtype=object) for i in range(len(documents))], dtype=object)
    # Generate word count by word by label
    vocabulary = list()
    for l in np.unique(labels):
        words_by_label = np.concatenate([documents[i][1] for i in range(len(documents)) if documents[i][0] == l])
        words_by_label = np.unique(words_by_label, return_counts=True)
        for i in range(len(words_by_label[0])):
            vocabulary.append([l, words_by_label[0][i], words_by_label[1][i]])

    return vocabulary


def compute(vocabulary, classes, sample_document, train_labels, to_stem):
    sample_doc_words = tokenize(sample_document, to_stem)  # A set of 'words' in the sample_document
    proxy_prob = np.empty(0)  # A set of proxies for the posterior prob(c_k | words)

    n = len(sample_doc_words)       # n - # of terms in sample_doc_words
    m = len(classes)                # m - # of classes

    class_probabilities = Counter(train_labels)

    # For each k-th class ck, compute the posterior prob(ck | sample_doc_words)
    # wi indicates the i-th term in sample_doc_words
    for k in range(m):
        # log_p_ck_S_proxy:
        # S is original string, but we are working with tokenized version W
        #       log p(ck | W) ∝ log p(ck) + SUM wi_count x log p(wi | ck)
        # Stuff to the right of ∝ is a proxy for log p(ck | W).
        # For =, would need to include p(W) which is a constant for a given W.
        # To avoid zero probability problem, use Laplace for log p(wi | ck)

        log_p_ck_S_proxy = 0

        vocab_ck = [v for v in vocabulary if v[0] == k]  # vocab_ck - Vocabulary from class ck documents

        # Probability of the k-th class ck in documents
        prob_ck = class_probabilities[k] * 1.0 / (np.sum(list(class_probabilities.values())))

        # For each term words[i] wi, compute the likelihood P(ck | wi)
        for i in range(n):
            # Obtain the count and probability of the term wi in the class ck documents
            # Laplacian P(wi|ck)
            laplace_p_wi_ck = get_laplace_prob_term(vocab_ck, sample_doc_words[i], m)

            log_p_ck_S_proxy += log_p(laplace_p_wi_ck)
#
        # Add the log prior log p(ck) so now have below.
        log_p_ck_S_proxy += log_p(prob_ck)

        # Append the proxy for posterior prob(ck | words) of the class ck to the array proxy_prob
        proxy_prob = np.append(proxy_prob, log_p_ck_S_proxy)

    # Derive reportable probabilities from the proxy ones
    # - take the max_value (the negative number closest to zero) from each to reduce underflow errors
    # - take exponent and normalise
    prob = proxy_prob - np.max(proxy_prob)
    prob = np.exp(prob)
    prob = prob/np.sum(prob)
    # Obtain an index of the class sample_class as the class in C,
    # having the maximum posterior prob(Ck | W)
    sample_class = np.where(prob == np.max(prob))[0][0]

    return prob, sample_class  # Return the array of posteriors prob and the index of sample_document sample_class


def evaluate(sample_data, sample_labels, model_vocabulary, classes, train_labels, args: NaiveBayesArgs):
    if args.verbose:
        print('Classification:')
        print('===============\n')

    # For each sample in sample_data, compute the class
    # Estimate the multinomial entropy
    correct_count = 0
    for i in range(len(sample_data)):

        pr_s = ' '
        prob, calculated_class = compute(model_vocabulary, classes, sample_data[i], train_labels, args)
        # Entropy: if p is 0 or 1, this ends up as 0 as no uncertainty
        # overall_entropy, individual_entropy = get_entropy(prob)

        for ci, p in zip(range(len(classes)), prob):
            pr_s += PROB_STATS_FMT % (classes[ci], p)

        correct_count += (int(sample_labels[i]) == calculated_class)

        if args.verbose:
            print(SAMPL_STATS_FMT
                  % (
                      sample_data[i],
                      classes[int(sample_labels[i])],
                      classes[calculated_class] if np.sum(prob) > 0 else 'None',
                      pr_s)
                  )
    return correct_count/len(sample_data)


def load_data(data_file_path: str, columns: list):
    """

    :param data_file_path: path to data file
    :param columns: pair of column headers; first is main data, second is a category or label
    :return:
    """
    df = pd.read_csv(data_file_path)
    df.head()

    return np.array(df[columns[0]]), np.array(df[columns[1]])


def stemming(tokenized_text: list, stemmer=nltk.LancasterStemmer()):
    """Stem stemmed_text by optionally chosen stemmer.

    :param tokenized_text: List of words to be stemmed; assumes already pre-processed.
    :param stemmer: NLTK stemmer (default LancasterStemmer)
    :return: List of stemmed words.
    """
    stemmed_text = [stemmer.stem(word) for word in tokenized_text]
    return stemmed_text


def tokenize(document: str, to_stem: bool = False):
    """
    Tokenize using NLTK library
    """
    punctuation = string.punctuation + "“”"
    for item in punctuation:
        document = document.replace(item, ' ')
    words = document.lower().split()

    stopwords = nltk.corpus.stopwords.words('english')
    words = [word for word in words if word not in stopwords and len(word) > 2]

    if to_stem:
        words = stemming(words)
    else:
        # Focus on nouns NN*, verbs VB* and adjectives JJ*
        words = [tag[0] for tag in nltk.pos_tag(words)
                 if re.match('NN', tag[1]) is not None
                 or re.match('JJ', tag[1]) is not None
                 or re.match('VB', tag[1]) is not None]

    return np.array([w for w in words])


def output_data(sample_set, documents, document_labels, classes, class_stats_format):
    print(MODEL_STATS_FMT % (len(classes), len(documents), len(sample_set)))

    print('Classes:')
    print('========\n')

    for k in range(len(classes)):
        documents_k = get_class_documents(documents, document_labels, k)
        p_ck = len(documents_k) / len(documents)  # P(k)
        pd_stats = class_stats_format % (len(documents_k), k, p_ck)
        print('C%d: %s %s' % (k, '{0: <12}'.format(classes[k]), pd_stats))

    print('\n')

    print('Documents:')
    print('==========\n')

    for i in range(len(documents)):
        print('C%d: \"%s...\"' % (i, documents[i][:80]))

    print("\n")


def save_vocab(save_dir: str, file_name: str, vocab: list):
    save_path = os.path.join(save_dir, file_name)
    # pickle - open in binary mode
    with open(save_path, 'wb') as outfile:
        pickle.dump(vocab, outfile)

    print('File saved: {}'.format(save_path))


PROB_STATS_FMT = 'Pr(%s) = %f '
CLASS_STATS_FMT = '[ Document count: %3d, P(C%d) = %f ]'
SAMPL_STATS_FMT = 'Document: [ \"%s\" ]\nTrue class: \"%s\" Calculated class: \"%s\" [%s]\n'
MODEL_STATS_FMT = '[ Classes: %d Documents: %d Samples: %d ]\n'
# MULTI_STATS_FMT = 'Multinomial Entropy: [ max: %f, real: %f ] classes/term\n'
LOG_ZERO = -99


def main():
    print('\nMultinomial Naїve Bayes\' classifier')
    print('====================================\n')

    args = get_arguments()

    train_data, train_labels = load_data(args.train_data_path, columns=['text', 'category'])
    eval_data, eval_labels = load_data(args.val_data_path, columns=['text', 'category'])

    classes, class_indices = load_data(args.classes_path, columns=['category', 'index'])

    classes_dict = {classes[i]: class_indices[i] for i in range(len(classes))}

    train_labels = [classes_dict[label] for label in train_labels]
    eval_labels = [classes_dict[label] for label in eval_labels]

    if args.verbose:
        output_data(eval_data, train_data, train_labels, classes, CLASS_STATS_FMT)

    for to_stem in [True, False]:
        print('----------------------------------------------------')
        print(f'STEMMING: {to_stem}')
        print('----------------------------------------------------')

        model_vocabulary = build_model(train_data, train_labels, to_stem)

        if args.to_save_model:
            file_name = 'model_vocabulary_stem.txt' if to_stem else 'model_vocabulary_no_stem.txt'
            file_name = f'{args.save_label}_{file_name}' if len(args.save_label) > 0 else file_name

            save_vocab(save_dir=args.output_dir, file_name=file_name, vocab=model_vocabulary)

        # Evaluate against training data
        train_accuracy = evaluate(train_data, train_labels, model_vocabulary, classes, train_labels, args)

        # Evaluate against validation data
        eval_accuracy = evaluate(eval_data, eval_labels, model_vocabulary, classes, train_labels, args)

        print(f'Training accuracy: {train_accuracy * 100:.1f}%')
        print(f'Validation accuracy: {eval_accuracy * 100:.1f}%')


if __name__ == '__main__':
    main()
