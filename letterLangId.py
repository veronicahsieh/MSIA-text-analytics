# Language Identification using letter bigrams

import sys
import pandas as pd
import numpy as np
import math


def get_txt_data(file_name):
    f = open(file_name, 'rb')
    text_data = f.read().decode('utf8', 'backslashreplace')
    f.close()
    return text_data


def get_letters(txt_data):
    """Takes the text data and returns a list of letters"""
    sentences = txt_data.split('\n')
    letters = [list(i.lower()) for i in sentences]
    letters = [y for word in letters for y in word]
    return letters


def create_letter_dict(letters_list):
    """Gets the counts for each letter (unigram)"""
    letters_df = pd.DataFrame(letters_list, columns=['letter'])
    letters_count = letters_df['letter'].value_counts()
    letter_dict = letters_count.to_dict()
    return letter_dict


# Create dictionary of bigrams and their counts
def create_bigram_dict(letters_list):
    """Takes a list of letters and returns a bigram dictionary with all bigrams and counts"""
    letters_length = len(letters_list)
    n = 2
    letters_bigram = []
    for i in range(0, letters_length - n + 1):
        n_gram = ''.join(letters_list[i: i + n])
        letters_bigram.append(n_gram)

    bigram_df = pd.DataFrame(letters_bigram, columns=['bigram'])
    bigram_count = bigram_df['bigram'].value_counts()
    bigram_dict = bigram_count.to_dict()
    return bigram_dict


# Calculate sentence probabilities
def get_bigrams(letters_list):
    """Get a list of a bigrams from a list of letters"""
    list_length = len(letters_list)
    n = 2
    list_of_bigrams = []
    for i in range(0, list_length - n + 1):
        n_gram = ''.join(letters_list[i: i + n])
        list_of_bigrams.append(n_gram)
    return list_of_bigrams


# Function to get probability for a sentence - no smoothing added

def sentence_prob(sentence_bigram, bigram_dict, letter_dict):
    """Takes a list of bigrams for a sentence and returns the bigram probability"""
    bigram_test_prob = []
    for bigram in sentence_bigram:  # bigram is list of bigrams for each sentence in test set
        letter1 = bigram[0]
        letter2 = bigram[1]
        if bigram in bigram_dict:  # if bigram exists, calculate probability
            bigram_prob = bigram_dict.get(bigram)/letter_dict.get(letter1)
            bigram_prob_pair = (bigram, bigram_prob)
            bigram_test_prob.append(bigram_prob)
        else:  # if bigram doesn't exist, add bigram to the dictionary
            bigram_dict[bigram] = 1
            if letter1 in letter_dict:
                bigram_prob = bigram_dict.get(bigram)/letter_dict.get(letter1)
                bigram_prob_pair = (bigram, bigram_prob)
                bigram_test_prob.append(bigram_prob)
            else:
                letter_dict[letter1] = 1
                bigram_prob = bigram_dict.get(bigram)/len(letter_dict)
                bigram_prob_pair = (bigram, bigram_prob)
                bigram_test_prob.append(bigram_prob)
    sentence_prob = math.exp(sum(np.log(bigram_test_prob)))
    return sentence_prob


def output_language(lol_bigrams):
    """Takes lists of bigram lists and outputs a list with the predicted language for each sentence"""
    line_id = 1
    test_answers = []
    for line in test_bigrams:
        language_prob = []
        eng_prob = sentence_prob(line, eng_bigram_dict, eng_letter_dict)
        italian_prob = sentence_prob(line, italian_bigram_dict, italian_letter_dict)
        french_prob = sentence_prob(line, french_bigram_dict, french_letter_dict)
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


# Calculate the accuracy using letter bi-grams

def calc_accuracy(predicted_list, actual_list):
    """Compares the solutions file with the model's predicted languages and returns accuracy"""
    num_correct = 0
    for i in range(len(actual_list)-1):
        if actual_list[i] == predicted_list[i]:
            num_correct += 1
    accuracy = num_correct/(len(actual_list)-1)
    accuracy_format = round(accuracy, 4) * 100
    answer = "Accuracy: {} %".format(accuracy_format)
    return answer


# Add-one smoothing functions
def add_one_prob(sentence_bigram, bigram_dict, letter_dict):
    """Takes a list of bigrams for a sentence and returns the bigram probability using add-one smoothing"""
    bigram_test_prob = []
    for bigram in sentence_bigram:  # bigram test is list of bigrams for FIRST sentence in test set
        letter1 = bigram[0]
        letter2 = bigram[1]
        if bigram in bigram_dict:  # if bigram exists, calculate probability
            bigram_prob = (bigram_dict.get(bigram)+1)/(letter_dict.get(letter1) + len(letter_dict))
            bigram_test_prob.append(bigram_prob)
        else:  # if bigram doesn't exist, add bigram to the dictionary
            bigram_dict[bigram] = 1
            if letter1 in letter_dict:
                bigram_prob = (bigram_dict.get(bigram)+1) / \
                    (letter_dict.get(letter1) + len(letter_dict))
                bigram_test_prob.append(bigram_prob)
            else:
                letter_dict[letter1] = 1
                bigram_prob = (bigram_dict.get(bigram)+1) / \
                    (letter_dict.get(letter1) + len(letter_dict))
                bigram_test_prob.append(bigram_prob)
    sentence_prob = math.exp(sum(np.log(bigram_test_prob)))
    return sentence_prob


def output_add_one_language(lol_bigrams):
    """Takes lists of bigram lists and outputs a list with the predicted language for each sentence"""
    line_id = 1
    test_answers = []
    for line in lol_bigrams:
        language_prob = []
        eng_prob = add_one_prob(line, eng_bigram_dict, eng_letter_dict)
        italian_prob = add_one_prob(line, italian_bigram_dict, italian_letter_dict)
        french_prob = add_one_prob(line, french_bigram_dict, french_letter_dict)
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

    # split test file into sentences
    solution_lines = solution.split('\n')

    english_data = get_txt_data('LangId.train.english')
    italian_data = get_txt_data('LangId.train.Italian')
    french_data = get_txt_data('LangId.train.French')

    # Create bigram and letter vocabulary for each training set
    eng_letters = get_letters(english_data)
    french_letters = get_letters(french_data)
    italian_letters = get_letters(italian_data)

    # Create dictionary of letters (vocab) and their counts
    eng_letter_dict = create_letter_dict(eng_letters)
    french_letter_dict = create_letter_dict(french_letters)
    italian_letter_dict = create_letter_dict(italian_letters)

    eng_bigram_dict = create_bigram_dict(eng_letters)
    french_bigram_dict = create_bigram_dict(french_letters)
    italian_bigram_dict = create_bigram_dict(italian_letters)

    # Get letter unigrams for test set
    test_sentence = test.split('\n')
    test_letters = [list(i.lower()) for i in test_sentence]

    # Get letter bigrams for test set
    test_bigrams = []
    for bigram in test_letters:
        bigram_test = get_bigrams(bigram)
        test_bigrams.append(bigram_test)

    # Outputting the predicted language for each sentence
    letter_guess = output_add_one_language(test_bigrams)
    letter_guess_accuracy = calc_accuracy(letter_guess, solution_lines)
    print(letter_guess_accuracy)

    with open('letterLangId.out', 'w') as f:
        for item in letter_guess:
            f.write("%s\n" % item)
