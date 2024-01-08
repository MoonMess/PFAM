import argparse
from utils import tools
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
dataset_folder = "random_split"


def data_preprocessing_propythia(sampling = True):
    #loading data 
    # from dataset_folder/train/, dataset_folder/test/ and dataset_folder/dev/
    print("Loading PFAM data")
    dataset_train = tools.read_data("train")
    dataset_test = tools.read_data("test")
    dataset_dev = tools.read_data("dev")
    if sampling :
        #sampling the data
        print("Sampling the data")
        dataset_train, dataset_dev, dataset_test = tools.sample_data(dataset_train, dataset_dev, dataset_test)
    #remove duplication 
    dataset_train, dataset_dev, dataset_test = tools.remove_duplication(dataset_train, dataset_dev, dataset_test)
    #label encoder
    dataset_train, dataset_dev, dataset_test = tools.label_encoder(dataset_train, dataset_dev, dataset_test)
    #sequence cleaning
    descriptors_train = tools.sequence_preprocessing(dataset_train)
    descriptors_test = tools.sequence_preprocessing(dataset_test)
    descriptors_dev = tools.sequence_preprocessing(dataset_dev)
    #features extraction
    features_train = tools.get_features(descriptors_train)
    features_test = tools.get_features(descriptors_test)
    features_dev = tools.get_features(descriptors_dev)
    #cleaning the new columns created
    #features_train, features_dev, features_test = tools.clean_features(features_train, features_dev, features_test)
    X_train, y_train = tools.split_X_y(features_train)
    X_test, y_test = tools.split_X_y(features_test)
    X_dev, y_dev = tools.split_X_y(features_dev)
    # Standardize features by normalizing each feature
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_dev = scaler.transform(X_dev)
    return X_train, y_train, X_dev, y_dev, X_test, y_test



def sequence_encoder(dataset_train, dataset_dev, dataset_test):
    #label encoder
    dataset_train, dataset_dev, dataset_test = tools.label_encoder(dataset_train, dataset_dev, dataset_test)

    #splitting the data into features and target
    [features, target] = tools.features_target_split(dataset_train)
    [features_test, target_test] = tools.features_target_split(dataset_test)
    [features_dev, target_dev] = tools.features_target_split(dataset_dev)

    #cutting the AA sequence into list of AA
    features = tools.cut_protein_sequence(features)
    features_test = tools.cut_protein_sequence(features_test)
    features_dev = tools.cut_protein_sequence(features_dev)

    #tokenizing the data
    print("Tokenizing the data")
    tokenizer_seq = Tokenizer(num_words = 2000)
    #fitting the tokenizer on the train set (ONLY)
    tokenizer_seq.fit_on_texts(features)
    #transforming the data
    print("Transforming the data")
    features = tokenizer_seq.texts_to_sequences(features)
    features_dev = tokenizer_seq.texts_to_sequences(features_dev)
    features_test = tokenizer_seq.texts_to_sequences(features_test)

    #padding the data
    print("Padding the data")
    #We are using aligned sequence, so we will have a max length of 500 to get the maximum information
    #500 ask for too much computin power when training. I will stop at 250
    features = pad_sequences(features, maxlen = 250, padding = "post", truncating = "post")
    features_test = pad_sequences(features_test, maxlen = features.shape[1], padding = "post", truncating = "post")
    features_dev = pad_sequences(features_dev, maxlen = features.shape[1], padding = "post", truncating = "post")
    vocabulary = len(tokenizer_seq.word_index)

    return features, target, features_dev, target_dev, features_test, target_test


def data_preprocessing_sequence_encoder(sampling = True):
    #loading data 
    # from dataset_folder/train/, dataset_folder/test/ and dataset_folder/dev/
    print("Loading PFAM data")
    dataset_train = tools.read_data("train")
    dataset_test = tools.read_data("test")
    dataset_dev = tools.read_data("dev")

    if sampling:
        print("Sampling the data")
        dataset_train, dataset_dev, dataset_test = tools.sample_data(dataset_train, dataset_dev, dataset_test)

    features, target, features_dev, target_dev, features_test, target_test = sequence_encoder(dataset_train, dataset_dev, dataset_test)

    return features, target, features_dev, target_dev, features_test, target_test

def saving_data(path, features, target, features_dev, target_dev, features_test, target_test):
    print("Saving data")
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(path+"/features.npy", features)
    target.to_csv(path+"/target.csv")
    np.save(path+"/features_dev.npy", features_dev)
    target_dev.to_csv(path+"/target_dev.csv")
    np.save(path+"/features_test.npy", features_test)
    target_test.to_csv(path+"/target_test.csv")

if __name__ == "__main__":

    ### preprocessing and saving
    #full dataset
    #1st method: sequence encoder

    #TIME CONSUMING and not used in my experiment (I used the sample only)
    # print("Preprocessing data for sequence encoder on the full dataset")
    # features, target, features_dev, target_dev, features_test, target_test = data_preprocessing_sequence_encoder(sampling = False)
    # if not os.path.exists("preprocessed_data"):
    #     os.makedirs("preprocessed_data")
    # path = "preprocessed_data/full/sequence_encoder/"
    # saving_data(path, features, target, features_dev, target_dev, features_test, target_test)
    
    # #2nd method: feature generation (propythia)
    # print("Preprocessing data for propythia on the full dataset")
    # features, target, features_dev, target_dev, features_test, target_test = data_preprocessing_propythia(sampling = False)
    # if not os.path.exists("preprocessed_data/full/propythia/"):
    #     os.makedirs("preprocessed_data/full/propythia/")
    # path = "preprocessed_data/full/propythia/"
    # saving_data2(path, features, target, features_dev, target_dev, features_test, target_test)


    #sample dataset
    #2nd method: feature generation (propythia)
    print("Preprocessing data for propythia on the sample dataset")
    features, target, features_dev, target_dev, features_test, target_test = data_preprocessing_propythia(sampling = True)
    path = "preprocessed_data/sample/propythia/"
    saving_data(path, features, target, features_dev, target_dev, features_test, target_test)
    
    #sequence encoder
    print("Preprocessing data for sequence encoder on the sample dataset")
    features, target, features_dev, target_dev, features_test, target_test = data_preprocessing_sequence_encoder(sampling = True)
    path = "preprocessed_data/sample/sequence_encoder/"
    saving_data(path, features, target, features_dev, target_dev, features_test, target_test)


