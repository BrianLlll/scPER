#!/usr/bin/env python
# coding: utf-8
## Author: Bingrui Li
## Date: Mar 22 2023
### Script for training adversarial deconfounding autoencoder 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.utils.class_weight import compute_class_weight
from numpy.random import seed
import tensorflow
import keras as ke
import keras.backend as K
from keras.layers import Input, Dense, Dropout
from keras.models import Model
create_gif = False
import sys
import argparse
from ADAE import AdversarialConfounderAutoencoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser("inputs")
    parser.add_argument("scRNA", help="Imputed scRNA-seq data from preprocess step.")
    parser.add_argument("label", help="labels for the scRNA-seq data that need to remove the bias")
    parser.add_argument("bulk", help="Expression matrix for bulk samples")
    args = parser.parse_args()


    print(f"sklearn: {sk.__version__}")
    print(f"pandas: {pd.__version__}")
    print(f"keras: {ke.__version__}")


#Read single cell reference data

    input_df = pd.read_csv(args.scRNA, index_col=0).T
    input_df = pd.DataFrame(input_df.values, index = input_df.index.astype(str))


###onehot encoding of batch label
    pheno_df=pd.read_csv(args.label, index_col = 0)
    pheno_df=pheno_df.dropna()
    pheno_df=pheno_df['Batch']
    pheno_type=len(pheno_df.unique().tolist())
    pheno_df=pd.get_dummies(pheno_df)


    X = input_df.sort_index()
    Z = pheno_df.sort_index()

    print("X ", X.shape)
    print("Y ", Z.shape)

#Train and test splits
    X_train, X_test, Z_train, Z_test = train_test_split(X, Z, test_size=0.2, random_state=12345)

#Standardize the data
    scaler = StandardScaler().fit(X_train)
    scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
    X_train = X_train.pipe(scale_df, scaler) 
    X_test = X_test.pipe(scale_df, scaler)

    print("X ", X_train.shape)
    print("Y ", Z_train.shape)

        
    adv_model = AdversarialConfounderAutoencoder(n_features=X_train.shape[1], pheno_n=pheno_type,
                                       latent_dim = 100, lambda_val = 1.0, random_seed = 1)

# pre-train both adverserial and classifier networks
    adv_model.pretrain(X_train, Z_train, 
                        validation_data=(X_test, Z_test), epochs=10)

# adverserial train on train set and validate on test set
    adv_model.fit(X_train, Z_train, 
            validation_data=(X_test, Z_test),
            T_iter = 10)

#Generate embedding for single cells
    embedding = adv_model._encoder.predict(X)
    embedding_df = pd.DataFrame(embedding, index = X.index)
    embedding_df.to_csv('/results/ADAE_' + str(100) + '_latents' + '.tsv',sep='\t')


##embeding for bulk samples
    bulk_df = pd.read_csv(args.bulk, index_col=0).T


    encoded_bulk_df = adv_model._encoder.predict(bulk_df)
    encoded_bulk_df = pd.DataFrame(encoded_bulk_df, index=bulk_df.index)

    encoded_bulk_df.columns.name = 'latent'
    encoded_bulk_df.columns = encoded_bulk_df.columns + 1
    encoded_bulk_df.T.to_csv('/results/bulk_100_latents.tsv', sep='\t')


#Record models
    model_json = adv_model._encoder.to_json()
    with open('/results/ADE_encoder_SC' + str(100) + '_Latents'+'.json', "w") as json_file:
        json_file.write(model_json)
    adv_model._encoder.save_weights('/results/ADE_encoder_SC' + str(100) + '_Latents'+'.h5')
    print("Saved model to disk")

    model_json = adv_model._decoder.to_json()
    with open('/results/ADE_decoder_SC' + str(100) + '_Latents'+'.json', "w") as json_file:
        json_file.write(model_json)
    adv_model._decoder.save_weights('/results/ADE_decoder_SC' + str(100) + '_Latents'+'.h5')
    print("Saved model to disk")

