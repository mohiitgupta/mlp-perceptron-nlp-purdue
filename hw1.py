import numpy as np
np.set_printoptions(threshold='nan')
import argparse
import nltk
import math
import re
import random
import scipy
import pickle
import sys

from collections import Counter
from nltk.stem.snowball import SnowballStemmer as Stemmer
from scipy.special import expit
# from stemming.porter2 import stem

stemmer = Stemmer("english")

dictionary_counts = dict()
dictionary_indices = dict()
total_documents = 0
regex = re.compile('[^a-zA-Z ]')

def sigmoid(x):
    return expit(x)

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def read_file(filename):
    with open(filename, 'rb') as f:
        weight = pickle.load(f)
    return weight

class Classifier(object):
    def __init__(self):
        pass

    def train():
        """
        Override this method in your class to implement train
        """
        raise NotImplementedError("Train method not implemented")

    def inference():
        """
        Override this method in your class to implement inference
        """
        raise NotImplementedError("Inference method not implemented")

class MLP(Classifier):
    """
    Implement your MLP here
    """
    weight_matrix = []#read_file("weight_matrix_mlp_file")
    number_hidden_nodes = 9
    hidden_weight_vector = []#read_file("hidden_weight_vector_mlp_file")
    def train(self, data, labels, learning_rate, hidden_learning_rate
        # , validation_data, validation_labels
        ):
        feature_matrix = []
        
        for line in data:
            line = regex.sub('',line)
            sentence_feature_vector = feature_extractor(line, dictionary_indices)
            feature_matrix.append(sentence_feature_vector)

        #initialize weight vectors
        for hidden_node in range(self.number_hidden_nodes):
            # weight_vector = np.zeros(len(dictionary_counts))
            weight_vector = np.random.rand(len(dictionary_counts))
            self.weight_matrix.append(weight_vector)

        self.hidden_weight_vector = np.random.rand(self.number_hidden_nodes)
        hidden_feature_vector = np.zeros(self.number_hidden_nodes)

        # print "feature matrix is ", feature_matrix


        iteration = 0
        sum_error = 0
        count_1_label = 0
        count_0_label = 0
        tries = 0
        previous_sum_error = 0
        previous_f_score = 0
        f_score = 0
        while True:
            iteration += 1
            
            # for label_index, feature_vector in enumerate(feature_matrix):
            label_index = random.randint(0, len(feature_matrix) - 1)
            feature_vector = feature_matrix[label_index]
            hidden_dot_product_array = np.zeros(self.number_hidden_nodes)
            for hidden_node in range(self.number_hidden_nodes):
                # print "weight vector ", weight_matrix[hidden_node]
                weight_vector = self.weight_matrix[hidden_node]
                # print "sentence vector is ", vector
                # print "weight vector is ", weight_vector
                # print "feature_vector is", feature_vector
                hidden_dot_product = np.dot(weight_vector, feature_vector)
                # print "hidden dot product for " + str(hidden_node) + " is " + str(hidden_dot_product)
                hidden_dot_product_array[hidden_node] = hidden_dot_product
                hidden_feature_vector[hidden_node] = sigmoid(hidden_dot_product_array[hidden_node])

            dot_product = np.dot(hidden_feature_vector, self.hidden_weight_vector)
            # print "dot product is ", dot_product
            output = sigmoid(dot_product)

            output_binary = 0
            if output > 0.5:
                count_1_label += 1
                output_binary = 1
            else:
                count_0_label += 1
                output_binary = 0
            # print "output is ", str(output) + " label is ", str(labels[label_index]) + " output_binary ", output_binary
            error = labels[label_index] - output_binary
            sum_error += error**2
            absolute_error = labels[label_index] - output

            # print "absolute error is ", absolute_error

            # print "weight_matrix is ", weight_matrix
            hidden_delta_vector = np.multiply(hidden_learning_rate * absolute_error
                * sigmoid_derivative(dot_product), hidden_feature_vector)
            self.hidden_weight_vector = self.hidden_weight_vector + hidden_delta_vector 
            + 0.0001 * self.hidden_weight_vector 
            # print "hidden weight_vector is ", hidden_weight_vector
            # print "delta_vector is ", delta_vector

            #update input weight vectors
            weight_matrix = []
            for i, weight_vector in enumerate(self.weight_matrix):
                delta_vector = np.multiply(learning_rate * absolute_error
                    * sigmoid_derivative(dot_product) * self.hidden_weight_vector[i]
                    * sigmoid_derivative(hidden_dot_product_array[i]), feature_vector)
                weight_vector = weight_vector + delta_vector 
                + 0.0001 * weight_vector
                weight_matrix.append(weight_vector)

            self.weight_matrix = weight_matrix
            
            if iteration > 4500:
                tries += 1
                iteration -= 4500
                # print '0 labels are ' + str(count_0_label) + " 1 labels are " + str(count_1_label)
                # print "sum error is ", sum_error
                # print "weight_matrix is ", self.weight_matrix[0]
                # print "hidden weight vector is ", self.hidden_weight_vector

                # validation_scores = self.validation(validation_data, validation_labels)
                # f_score = validation_scores[2] 
                # print "f score is ", f_score

                # if tries >= 30 or ((previous_sum_error - sum_error) < 50 and f_score > 0.68 and (f_score - previous_f_score) < 0.01):
                #     print "training complete mlp..."
                #     # weight_vector.np.matrix.dump("weight_vector")
                #     with open('weight_matrix_mlp_file', 'wb') as f:
                #         pickle.dump(self.weight_matrix, f)

                #     with open('hidden_weight_vector_mlp_file', 'wb') as f:
                #         pickle.dump(self.hidden_weight_vector, f)

                #     print "dumping weight vector to file complete...."
                #     break

                if tries >= 21:
                    break

                previous_sum_error = sum_error
                previous_f_score = f_score
                sum_error = 0
                count_1_label = 0
                count_0_label = 0
                sys.stdout.flush() 

    def inference(self, data):
        output_labels = []
        count_1_label = 0
        count_0_label = 0
        i = 1
        # print "weight matrix is ", self.weight_matrix
        # print "hidden nodes is ", self.number_hidden_nodes
        # print "hidden weight vector is ", self.hidden_weight_vector

        for i, sentence in enumerate(data):
            hidden_feature_vector = np.zeros(self.number_hidden_nodes)
            sentence = regex.sub('',sentence)
            feature_vector = feature_extractor(sentence, dictionary_indices)
            hidden_dot_product_array = np.zeros(self.number_hidden_nodes)
            for hidden_node in range(self.number_hidden_nodes):
                # print "weight vector ", weight_matrix[hidden_node]
                weight_vector = self.weight_matrix[hidden_node]
                # print "sentence vector is ", vector
                # print "weight vector is ", weight_vector
                # print "feature_vector is", feature_vector
                hidden_dot_product = np.dot(weight_vector, feature_vector)
                # print "hidden dot product for " + str(hidden_node) + " is " + str(hidden_dot_product)
                hidden_dot_product_array[hidden_node] = hidden_dot_product
                hidden_feature_vector[hidden_node] = sigmoid(hidden_dot_product_array[hidden_node])

            dot_product = np.dot(hidden_feature_vector, self.hidden_weight_vector)
            # print "dot product is ", dot_product
            output = sigmoid(dot_product)
            output_binary = 0
            if output > 0.5:
                count_1_label += 1
                output_binary = 1
            else:
                count_0_label += 1
                output_binary = 0

            output_labels.append(output_binary)

        # print "value of count 1 label MLP ", count_1_label
        # print "value of count 0 label MLP ", count_0_label
        if (count_1_label == 0):
            output_labels[i-1] = 1
        if (count_0_label == 0):
            output_labels[i-2] = 0

        return output_labels

    def validation(self, validation_data, validation_labels):
        
        output_labels = self.inference(validation_data)
        validation_scores = evaluate(output_labels, validation_labels)

        print "validation scores", validation_scores

        return validation_scores

    def __init__(self):
        super(MLP, self).__init__()
        pass



class Perceptron(Classifier):
    """
    Implement your Perceptron here
    """
    
    
    weight_vector = []#read_file("weight_vector_perceptron_file")
    def train(self, data, labels, learning_rate
        # , validation_data, validation_labels
        ):
        feature_matrix = []
        # create_dictionary(data)
        
        for line in data:
            line = regex.sub('',line)
            sentence_feature_vector = feature_extractor(line, dictionary_indices)
            feature_matrix.append(sentence_feature_vector)
        
        # print "feature matrix is ", feature_matrix[0]
        self.weight_vector = np.random.rand(len(dictionary_counts))
        # self.weight_vector = np.zeros(len(dictionary_counts))
        # print "length of dictionary is " + str(len(dictionary_counts)) + " " + str(len(dictionary_indices))
        # print "weight vector is ", weight_vector
        # print "feature matrix length ", len(feature_matrix)
        # array = np.array(dictionary.keys())
        # print "dictionary ", dictionary_indices['content']
        iteration = 0
        count_1_label = 0
        count_0_label = 0
        sum_error = 0
        previous_sum_error = 0
        previous_f_score = 0
        f_score = 0
        tries = 0
        while True:
            label_index = random.randint(0, len(feature_matrix) - 1)
            # for label_index, feature_vector in enumerate(feature_matrix):
            iteration += 1
            # print "sentence vector is ", vector
            dot_product = np.dot(feature_matrix[label_index], self.weight_vector)
            # print "feature vector ", feature_vector
            output = sigmoid(dot_product)
            output_binary = 0
            if output > 0.5:
                output_binary = 1
                count_1_label += 1
            else:
                output_binary = 0
                count_0_label += 1
            # print "output is ", str(output) + " label is ", str(labels[label_index]) + " output_binary ", output_binary
            error = labels[label_index] - output_binary
            sum_error += error**2
            delta_vector = np.multiply(learning_rate * (labels[label_index] - output), feature_matrix[label_index])
            self.weight_vector = self.weight_vector + delta_vector + 0.00001 * self.weight_vector
            # print "iteration ", iteration
            
            if iteration > 1000:
                tries += 1
                # print "dot product is ", dot_product
                # print "weight_vector is ", self.weight_vector
                # print "iteration is ", iteration
                # print "count_1_label is ", count_1_label
                # print "count_0_label is ", count_0_label
                # print "delta_vector is ", delta_vector
                # print "sum error is ", sum_error    
                # print "weight_vector ", weight_vector
                # validation_scores = self.validation(validation_data, validation_labels)
                # f_score = validation_scores[2] 
                # print "f score is ", f_score

                # if tries >= 30 or ((previous_sum_error - sum_error) < 50 and f_score > 0.70 and (f_score - previous_f_score) < 0.01):
                    # print "training complete..."
                    # # weight_vector.np.matrix.dump("weight_vector")
                    # with open('weight_vector_perceptron_file', 'wb') as f:
                    #     pickle.dump(self.weight_vector, f)
                    # print "dumping weight vector to file complete...."
                    # break
                if tries == 18:
                    # print "training complete..."
                    # # weight_vector.np.matrix.dump("weight_vector")
                    # with open('weight_vector_perceptron_file', 'wb') as f:
                    #     pickle.dump(self.weight_vector, f)
                    # print "dumping weight vector to file complete...."
                    break

                previous_sum_error = sum_error
                previous_f_score = f_score
                sum_error = 0
                count_1_label = 0
                count_0_label = 0
                iteration -= 1000
                sys.stdout.flush()

    def inference(self, data):

        output_labels = []
        count_1_label = 0
        count_0_label = 0
        i = 0
        # print "weight vector is ", self.weight_vector
        for i, sentence in enumerate(data):
            sentence = regex.sub('',sentence)
            sentence_vector = feature_extractor(sentence, dictionary_indices)
            output_label = sigmoid(np.dot(sentence_vector, self.weight_vector))
            if output_label > 0.5:
                output_binary = 1
            else:
                output_binary = 0
            if output_binary == 1:
                count_1_label += 1
            elif output_binary == 0:
                count_0_label += 1

            output_labels.append(output_binary)

        # print "value of count 1 label ", count_1_label
        if (count_1_label == 0):
            output_labels[i-1] = 1
        if (count_0_label == 0):
            output_labels[i-2] = 0

        return output_labels

    def validation(self, validation_data, validation_labels):
        
        output_labels = self.inference(validation_data)
        validation_scores = evaluate(output_labels, validation_labels)

        print "validation scores", validation_scores

        return validation_scores


    def __init__(self):
        super(Perceptron, self).__init__()
        pass

def create_dictionary(data):
    index = 0
    global dictionary_counts
    global dictionary_indices
    global total_documents
    for line in data:
        total_documents += 1
        line = regex.sub('',line)
        words = line.split()
        buffer_dictionary = {}
        for word in words:
            buffer_dictionary[stemmer.stem(word)] = 1
        for key in buffer_dictionary:
            # print "word is ", word
            # word = stem(word)
            # print "stem is ", word
            if not key in dictionary_indices:
                dictionary_indices[key] = index
                dictionary_counts[key] = 1
                index += 1
            else:
                dictionary_counts[key] += 1
    # print "count dictionary is ", dictionary_counts
    # print "dictionary indices are ", dictionary_indices



def feature_extractor_using_tfidf(line, dictionary_indices):
    """
    implement your feature extractor here
    """
    feature_vector = np.zeros(len(dictionary_indices),)
    # print "feature vector length", len(feature_vector)
    count = 0
    words = line.split()
    for word in words:
        count += 1
        word = stemmer.stem(word)
        # word = stem(word)
        index = dictionary_indices.get(word, None)
        # print "dictionary index ", index
        if index != None:
            feature_vector[index] += 1

    
    for word in words:
        word = stemmer.stem(word)
        i = dictionary_indices.get(word, None)
        #use tf idf as weighting vector
        feature_vector[i] = feature_vector[i] * 1.0 / count * np.log(total_documents/ dictionary_counts[word] + 1)

    return feature_vector

def feature_extractor_without_tfidf(line, dictionary_indices):
    """
    implement your feature extractor here
    """
    feature_vector = np.zeros(len(dictionary_indices),)
    # print "feature vector length", len(feature_vector)
    min = 10000
    max = 0
    for word in line.split():
        word = stemmer.stem(word)
        # word = stem(word)
        index = dictionary_indices.get(word, None)
        # print "dictionary index ", index
        if index != None:
            feature_vector[index] += 1
            if feature_vector[index] > max:
                max = feature_vector[index]
            elif feature_vector[index] < min:
                min = feature_vector[index]
    
    for feature in feature_vector:
        feature = feature * 1.0 / (max - min + 1)
    # feature_vector = np.linalg.norm(feature_vector)
    return feature_vector

def feature_extractor(line, dictionary_indices):
    feature_vector = feature_extractor_using_tfidf(line, dictionary_indices)
    return feature_vector

def evaluate(preds, golds):
    tp, pp, cp = 0.0, 0.0, 0.0
    for pred, gold in zip(preds, golds):
        if pred == 1:
            pp += 1
        if gold == 1:
            cp += 1
        if pred == 1 and gold == 1:
            tp += 1
    # print "tp ", tp
    # print "pp ", pp
    # print "cp ", cp
    precision = tp / pp
    recall = tp / cp
    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        return (precision, recall, 0.0)
    return (precision, recall, f1)

def fetch_data(sentences_file, labels_file):
    with open(sentences_file) as f:
        data = f.readlines()
    with open(labels_file) as g:
        labels = [int(label) for label in g.read()[:-1].split("\n")]
    return data, labels
def main():

    argparser = argparse.ArgumentParser()
    
    dictionary_data, labels_dummy = fetch_data("whole_data.txt", "labels.txt")
    data, labels = fetch_data("sentences.txt", "labels.txt")
    data_training, labels_training = fetch_data("sentences_training.txt", "labels_training.txt")
    # validation_data, validation_labels = fetch_data("sentences_validation.txt", "labels_validation.txt")
    mymlp = MLP()
    myperceptron = Perceptron()
    create_dictionary(dictionary_data)

    """
    Training
    """
    learning_rate = 0.5
    hidden_learning_rate = 0.5
    perceptron_learning_rate = 0.15
    myperceptron.train(data_training, labels_training, perceptron_learning_rate
        # , validation_data, validation_labels
        )

    mymlp.train(data_training, labels_training, learning_rate, hidden_learning_rate
        # , validation_data, validation_labels
        )
    """
    Testing on testing data set
    """

    # with open("sentences_test.txt") as f:
    #     test_x = f.readlines()
    # with open("labels_test.txt") as g:
    #     test_y = np.asarray([int(label) for label in g.read()[:-1].split("\n")])

    """
    a numpy array of integers
    """
    # predicted_y = mymlp.inference(test_x)
    # precision, recall, f1 = evaluate(predicted_y, test_y)
    # print "MLP results", precision, recall, f1

    # predicted_y = myperceptron.inference(test_x)
    # precision, recall, f1 = evaluate(predicted_y, test_y)
    # print "Perceptron results", precision, recall, f1

    """
    Testing on unseen testing data in grading
    """
    argparser.add_argument("--test_data", type=str, default="../test_sentences.txt", help="The real testing data in grading")
    argparser.add_argument("--test_labels", type=str, default="../test_labels.txt", help="The labels for the real testing data in grading")

    parsed_args = argparser.parse_args(sys.argv[1:])
    real_test_sentences = parsed_args.test_data
    real_test_labels = parsed_args.test_labels
    with open(real_test_sentences) as f:
        real_test_x = f.readlines()
    with open(real_test_labels) as g:
        real_test_y = np.asarray([int(label) for label in g.read()[:-1].split("\n")])

    # print real_test_y

    predicted_y = mymlp.inference(real_test_x)
    # print predicted_y
    precision, recall, f1 = evaluate(predicted_y, real_test_y)
    print "MLP results", precision, recall, f1

    predicted_y = myperceptron.inference(real_test_x)
    precision, recall, f1 = evaluate(predicted_y, real_test_y)
    print "Perceptron results", precision, recall, f1



if __name__ == '__main__':
    main()
