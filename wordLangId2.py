
# Language Identification using word bigrams

# load the necessary packages
import sys
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


# Reading in the training & test data

def get_txt_data(file_name):
    f = open(file_name, 'rb')
    text_data = f.read().decode('utf8', 'backslashreplace')
    f.close()
    return text_data


def get_words(txt_data):
    """Takes the text data and returns a list of words; also adds denoters for beginning/end of sentence """
    sentences = txt_data.split('\n')  # list of sentences
    sentence_list = [i.lower().split(' ') for i in sentences]
    c = [a.insert(0, '<s>') for a in sentence_list]
    d = [z.append('</s>') for z in sentence_list]
    words = [word for sentence in sentence_list for word in sentence]
    return words

# Create bigram and word vocabulary for each training set


def get_words(txt_data):
    """Takes the text data and returns a list of words; also adds denoters for beginning/end of sentence """
    sentences = txt_data.split('\n')  # list of sentences
    sentence_list = [i.lower().split(' ') for i in sentences]
    c = [a.insert(0, '<s>') for a in sentence_list]
    d = [z.append('</s>') for z in sentence_list]
    words = [word for sentence in sentence_list for word in sentence]
    return words


def get_word_bigrams(letters_list):
    """Get a list of a bigrams from a list of letters"""
    list_length = len(letters_list)
    n = 2
    list_of_bigrams = []
    for i in range(0, list_length - n + 1):
        n_gram = ' '.join(letters_list[i: i + n])
        list_of_bigrams.append(n_gram)
    return list_of_bigrams


def create_word_dict(words_list):
    """Gets the counts for each letter"""
    words_df = pd.DataFrame(words_list, columns=['letter'])
    words_count = words_df['letter'].value_counts()
    words_dict = words_count.to_dict()
    return words_dict


def create_word_bigram_dict(words_list):
    """Takes a list of letters and returns a bigram dictionary with all bigrams and counts"""
    words_length = len(words_list)
    n = 2
    words_bigram = []
    for i in range(0, words_length - n + 1):
        n_gram = ' '.join(words_list[i: i + n])
        words_bigram.append(n_gram)

    bigram_df = pd.DataFrame(words_bigram, columns=['bigram'])
    bigram_count = bigram_df['bigram'].value_counts()
    bigram_dict = bigram_count.to_dict()
    return bigram_dict


def word_sentence_prob(sentence_bigram, bigram_dict, letter_dict):
    """Takes a list of bigrams for a sentence and returns the bigram probability"""
    bigram_test_prob = []
    for bigram in sentence_bigram:  # bigram test is list of bigrams for FIRST sentence in test set
        bigram_words = bigram.split(' ')
        word1 = bigram_words[0]
        word2 = bigram_words[1]
        if bigram in bigram_dict:  # if bigram exists, calculate probability
            bigram_prob = (bigram_dict.get(bigram)+1)/(letter_dict.get(word1) + len(letter_dict))
            bigram_test_prob.append(bigram_prob)
        else:  # if bigram doesn't exist, add bigram to the dictionary
            bigram_dict[bigram] = 1
            if word1 in letter_dict:
                bigram_prob = bigram_dict.get(bigram)/(letter_dict.get(word1) + len(letter_dict))
                bigram_test_prob.append(bigram_prob)
            else:
                letter_dict[word1] = 1
                bigram_prob = bigram_dict.get(
                    bigram)/(letter_dict.get(word1) + len(letter_dict))  # letter_dict.get(letter1)
                bigram_test_prob.append(bigram_prob)
    sentence_prob = math.exp(sum(np.log(bigram_test_prob)))
    return sentence_prob


def get_test_bigrams(test_sentence_words):
    """Returns a list of bigram lists for each sentence in test set"""
    test_word_bigrams = []
    for word in test_words:
        bigram_word_test = get_word_bigrams(word)
        test_word_bigrams.append(bigram_word_test)
    return test_word_bigrams


def output_language_words(lol_bigrams):
    """Takes list of bigram lists and outputs a list with the predicted language for each sentence"""
    line_id = 1
    test_answers = []
    for line in lol_bigrams:
        language_prob = []
        eng_prob = word_sentence_prob(line, eng_word_bigram_dict, eng_word_dict)
        italian_prob = word_sentence_prob(line, italian_word_bigram_dict, italian_word_dict)
        french_prob = word_sentence_prob(line, french_word_bigram_dict, french_word_dict)
        language_prob.extend([eng_prob, italian_prob, french_prob])
        answer = max(language_prob)
        language_id = language_prob.index(answer)
        if language_id == 0:
            language = 'English'
        elif language_id == 1:
            language = 'Italian'
        elif language_id == 2:
            language = 'French'
        language
        line_output = str(line_id) + ' ' + language
        test_answers.append(line_output)
        line_id += 1
    return test_answers


def calc_accuracy(predicted_list, actual_list):
    """Compares the solutions file with the model's predicted languages and returns accuracy"""
    num_correct = 0
    for i in range(len(actual_list)-1):
        if actual_list[i] == predicted_list[i]:
            num_correct += 1
    accuracy = num_correct/(len(actual_list)-1)
    accuracy_format = round(accuracy, 4) * 100
    answer = "Accuracy: {} %".format(accuracy_format)
    print(num_correct)
    return answer


# Functions for Good Turing smoothing

def get_count_of_counts(word_bigram_dict):
    """Takes bigram dictionary and returns df with number of bigrams for each frequency"""
    word_bigram_df = pd.DataFrame.from_dict(
        word_bigram_dict, orient='index').rename(columns={0: 'n'}).reset_index()
    word_bigram_freq = word_bigram_df.groupby('n').count(
    ).reset_index().rename(columns={'index': 'num_of_words'})
    return word_bigram_freq


def good_turing_prob(sentence_bigram, bigram_dict, bigram_freq_df):
    """Takes a list of bigrams for one sentence and returns the bigram probability using
    Good Turing smoothing"""
    gt_prob = []
    known = 0
    for bigram in sentence_bigram:
        known += 1
    unknown_count = len(sentence_bigram) - known

    for bigram in sentence_bigram:
        if bigram in bigram_dict:
            bigram_freq = bigram_dict.get(bigram)
            bigram_N_plus_one = bigram_freq_df[bigram_freq_df.n == bigram_freq+1]
            if len(bigram_N_plus_one) != 0:
                bigram_N_plus_one_int = int(bigram_N_plus_one.num_of_words)
                bigram_N = int(bigram_freq_df[bigram_freq_df.n == bigram_freq].num_of_words)
                gt_bigram_prob = (bigram_freq + 1) * bigram_N_plus_one_int / \
                    bigram_N * 1/sum(bigram_dict.values())
                gt_prob.append(gt_bigram_prob)
            else:
                bigram_N = int(bigram_freq_df[bigram_freq_df.n == bigram_freq].num_of_words)
                bigram_N_plus_one_int = (bigram_N + 1)/2
                gt_bigram_prob = (bigram_freq + 1) * bigram_N_plus_one_int / \
                    bigram_N * 1/sum(bigram_dict.values())
                gt_prob.append(gt_bigram_prob)

        else:  # if bigram is not dictionary, i.e. bigram count is 0
            zero_freq = unknown_count
            bigram_N_plus_one = bigram_freq_df[bigram_freq_df.n == 1]
            bigram_N = zero_freq
            gt_bigram_prob = 1/sum(bigram_dict.values())  # bigram_N_plus_one/bigram_N
            gt_prob.append(gt_bigram_prob)
    sentence_prob = math.exp(sum(np.log(gt_prob)))
    return sentence_prob


def output_good_turing_language(lol_bigrams):
    """Takes list of bigram lists and outputs a list with the predicted language for each sentence"""
    line_id = 1
    test_answers = []
    for line in lol_bigrams:
        language_prob = []
        eng_prob = good_turing_prob(line, eng_word_bigram_dict, eng_bigram_coc)
        italian_prob = good_turing_prob(line, italian_word_bigram_dict, italian_bigram_coc)
        french_prob = good_turing_prob(line, french_word_bigram_dict, french_bigram_coc)
        language_prob.extend([eng_prob, italian_prob, french_prob])
        answer = max(language_prob)
        language_id = language_prob.index(answer)
        if language_id == 0:
            language = 'English'
        elif language_id == 1:
            language = 'Italian'
        elif language_id == 2:
            language = 'French'
        language
        line_output = str(line_id) + ' ' + language
        test_answers.append(line_output)
        line_id += 1
    return test_answers


if __name__ == '__main__':

    # Read in test and solution files
    test_data = sys.argv[1]
    solution_data = sys.argv[2]
    test = get_txt_data(test_data)
    solution = get_txt_data(solution_data)

    # solution_lines is the list of solutions for the test file
    solution_lines = solution.split('\n')

    english_data = get_txt_data('LangId.train.english')
    french_data = get_txt_data('LangId.train.French')
    italian_data = get_txt_data('LangId.train.Italian')

    eng_words = get_words(english_data)
    french_words = get_words(french_data)
    italian_words = get_words(italian_data)

    # Word unigram and bigram for English, French, and Italian

    eng_word_dict = create_word_dict(eng_words)
    french_word_dict = create_word_dict(french_words)
    italian_word_dict = create_word_dict(italian_words)

    eng_word_bigram_dict = create_word_bigram_dict(eng_words)
    french_word_bigram_dict = create_word_bigram_dict(french_words)
    italian_word_bigram_dict = create_word_bigram_dict(italian_words)

    # Get a list of words for test set

    test_sentence = test.split('\n')
    test_sentence
    test_words = [i.lower().split(' ') for i in test_sentence]
    test_a = [a.insert(0, '<s>') for a in test_words]
    test_b = [z.append('</s>') for z in test_words]

    # Create bigrams from test sentences
    test_bigram_words = get_test_bigrams(solution_lines)

    # Find the number of words with n-frequency

    eng_bigram_coc = get_count_of_counts(eng_word_bigram_dict)
    italian_bigram_coc = get_count_of_counts(italian_word_bigram_dict)
    french_bigram_coc = get_count_of_counts(french_word_bigram_dict)

    # Calculating bigram probabilities using Good Turing smoothing

    test = output_good_turing_language(test_bigram_words)

    test_accuracy = calc_accuracy(test, solution_lines)
    print(test_accuracy)

    with open('wordLangId2.out', 'w') as f:
        for item in test:
            f.write("%s\n" % item)
