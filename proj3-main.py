# -*- coding: utf-8 -*-
# Project3 - SP22 - CS424/525
# Authors: Cayse Rogers, Hadeer Farahat

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
#from tensorflow.math import confusion_matrix
from sklearn.metrics import confusion_matrix
from PIL import Image
import glob
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


def feed_forward(lr,decay,momentum,n):
    # Creating feed forward neural network
    model = Sequential()
    model.add(layers.Dense(1024,activation = "tanh"))
    model.add(layers.Dense(512,activation = "sigmoid"))
    model.add(layers.Dense(100,activation = "relu"))
    model.add(layers.Dense(n,activation = "softmax"))
    opt = keras.optimizers.SGD(learning_rate=lr,decay=decay,momentum=momentum)
    model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
    
    return model

def cnn(lr,decay,momentum,n):
    # Creaing convolutional neural network
    model = Sequential()
    model.add(layers.Conv2D(40,5,input_shape = (32,32,1),activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(100,activation = "relu"))
    model.add(layers.Dense(n,activation = "softmax"))
    opt = keras.optimizers.SGD(learning_rate=lr,decay=decay,momentum=momentum)
    model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
    
    return model
    
def custom_cnn(lr,decay,momentum,n):
    # Creating custom convolutional network
    model = Sequential()
    model.add(layers.Conv2D(100,5,input_shape = (32,32,1),activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(1,1,activation="relu"))
    model.add(layers.Conv2D(50,2,activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(100,activation = "relu"))
    model.add(layers.Dense(50,activation = "relu"))
    model.add(layers.Dense(n,activation = "softmax"))
    opt = keras.optimizers.SGD(learning_rate=lr,decay=decay,momentum=momentum)
    model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
    model.summary()
    return model

def custom_both(lr,decay,momentum,n1,n2):
    ininputs = tf.keras.Input(shape=(32,32,1,))
    # Input layers
    in1 = layers.Conv2D(100,5,activation="relu")(ininputs)
    in1 = layers.MaxPooling2D(pool_size=(2, 2))(in1)
    in1 = layers.Conv2D(1,1,activation="relu")(in1)
    in1 = layers.Conv2D(50,5,activation="relu")(in1)
    in1 = layers.MaxPooling2D(pool_size=(2, 2))(in1)
    in1 = layers.Flatten()(in1)
    # Branch 1
    out1 = layers.Dense(100,activation = "relu")(in1)
    out1 = layers.Dense(n1,activation = "softmax")(out1)
    #Branch 2
    out2 = layers.Dense(100,activation = "relu")(in1)
    out2 = layers.Dense(n2,activation = "softmax")(out2)
    model = Model(inputs = ininputs,outputs = [out1,out2])
    opt = keras.optimizers.SGD(learning_rate=lr,decay=decay,momentum=momentum)
    model.compile(loss=['categorical_crossentropy','categorical_crossentropy'],
                  optimizer=opt,metrics=['accuracy'])
    return model
    
'''
The following code up until main is from an open source article on the Keras website:
    https://keras.io/examples/generative/vae/
'''
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def call(self,input):
        #added to fix validation error
        return self
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

def auto(latent_dim):
    #Build encoder
    #latent_dim = 10  #Latent dimension should be at least 5
    encoder_inputs = keras.Input(shape=(32, 32, 1))
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()

    #Build decoder
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(8 * 8 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((8, 8, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()


    vae = VAE(encoder, decoder)
    #opt = keras.optimizers.SGD(learning_rate=lr,decay=decay,momentum=momentum)
    #vae.compile(optimizer=opt)
    vae.compile(optimizer=keras.optimizers.Adam(),loss="mse")
    
    return vae


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="CS424/525 - Project 3")
    parser.add_argument("--task")
    parser.add_argument("--attribute")
    args = parser.parse_args()
    
    task = args.task
    att = args.attribute
    
    df_train = pd.read_csv("fairface_label_train.csv") 
    df_test = pd.read_csv("fairface_label_val.csv")    


    if att == "gender": n = 2
    elif att == "age": n = 9 
    elif att == "race": n = 7

    y_train = df_train.loc[:,att].to_numpy() # training labels
    y_train = np.reshape(y_train,(86744,1))

    y_test = df_test.loc[:,att].to_numpy() # testing labels
    y_test = np.reshape(y_test,(10954,1))

    # One-hot encoding of labels
    oh_encoder = OneHotEncoder(sparse=False)
    oh_encoder.fit(y_train)
    y_train = oh_encoder.transform(y_train)
    y_test = oh_encoder.transform(y_test)

    # Reading in training images
    i = np.arange(1,10)
    x_train = []
    for file in glob.glob("train/*.jpg"):
        im=Image.open(file)
        x_train.append(list(im.getdata()))
        im.close()
    x_train = np.array(x_train)
    
    # Reading in testing images
    x_test = []

    for file in glob.glob("val/*.jpg"):
        im=Image.open(file)
        x_test.append(list(im.getdata()))
        im.close()
    x_test = np.array(x_test)
    
    print("finished input reading")
    
    # Min-Max scaling of data
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    
    age_labels = ["0-2","3-9","10-19","20-29","30-39","40-49","50-59","60-69","more than 70"]
    gender_labels = ["Male","Female"]
    race_labels = ["East Asian",  "Indian" , "Black" , "White" ,"Middle Eastern" , "Southeast Asian","Latino_Hispanic"]
    if task == "both":
        if att == "gender":
            att2 = "age"
            n2 = 9
            n1labels = gender_labels
            n2labels = age_labels
        elif att == "age":
            att2 = "race"
            n2 = 7
            n1labels = age_labels
            n2labels = race_labels
        elif att == "race":
            att2 = "gender"
            n2 = 2
            n1labels = race_labels
            n2labels = gender_labels

        y2_train = df_train.loc[:,att2].to_numpy() # training labels
        y2_train = np.reshape(y2_train,(86744,1))

        y2_test = df_test.loc[:,att2].to_numpy() # testing labels
        y2_test = np.reshape(y2_test,(10954,1))
        
        # One-hot encoding of labels
        oh2_encoder = OneHotEncoder(sparse=False)
        oh2_encoder.fit(y2_train)
        y2_train = oh2_encoder.transform(y2_train)
        y2_test = oh2_encoder.transform(y2_test)
        
        y_train = [y_train,y2_train]
        y_test = [y_test,y2_test]
            
    lr = 0.05
    decay = 0
    momentum = 0.9

    if task == "ff":
        model = feed_forward(lr,decay,momentum,n)
    elif task == "cnn":
        print('x_train shape : ', x_train.shape)
        x_train = np.reshape(x_train,(86744,32,32,1))
        x_test = np.reshape(x_test,(10954,32,32,1))
        model = cnn(lr,decay,momentum,n)
    elif task == "custom":
        x_train = np.reshape(x_train,(86744,32,32,1))
        x_test  = np.reshape(x_test,(10954,32,32,1))
        model   = custom_cnn(lr,decay,momentum,n)
    elif task == "both":
        x_train = np.reshape(x_train,(86744,32,32,1))
        x_test  = np.reshape(x_test,(10954,32,32,1))
        model   = custom_both(lr,decay,momentum,n,n2)
    elif task == "auto":
        latent_dim = 10
        x_train = np.reshape(x_train,(86744,32,32))
        x_test  = np.reshape(x_test,(10954,32,32))
        # Structuring data for use in auto encoder
        data = np.concatenate([x_train, x_test], axis=0)
        data = np.expand_dims(data, -1).astype("float32") / 255     
        test_data = np.concatenate([x_test, x_test], axis=0)
        test_data = np.expand_dims(data, -1).astype("float32") / 255     
        x_train = np.expand_dims(x_train, -1).astype("float32") / 255 
        x_test  = np.expand_dims(x_test, -1).astype("float32") / 255         
        model = auto(latent_dim)

    # Fitting and testing
    print(task + " fitting start")
    if task != "auto":
        history = model.fit(x_train,y_train,epochs=100,batch_size=100)
        score = model.evaluate(x_test, y_test)
        y_pred = model.predict(x_test)
    else:   
        history = model.fit(data, epochs=10, batch_size=128)
        #history = model.fit(x_train, epochs=2, batch_size=128, validation_data=(x_test, None))
        np.random.shuffle(data)
        y_pred = model.encoder.predict(data[:10])
        #y_pred = model.decoder.predict(y_pred)

    keys = list(history.history.keys())
    
    if task != "both" and task != "auto":
        y_pred = np.argmax(y_pred,axis=1)
        y_test = np.argmax(y_test,axis=1)
        con = confusion_matrix(y_test,y_pred)
        # Accuracy plot
        plt.plot(history.history["accuracy"])
        plt.title(task + " Accuracy for " + att)
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.savefig(task + "Acc_" + att + ".png")
        plt.clf()
        # Loss plot
        plt.plot(history.history["loss"])
        plt.title(task + " Loss for " + att)
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.savefig(task + "Loss_" + att + ".png")
        plt.clf()
        # Confustion matrix heat map
        plt.imshow(con, cmap='hot', interpolation='nearest')
        plt.title(task + " Confusion for " + att)
        plt.savefig(task + "Con_" + att + ".png")
        plt.clf()
        
        print(score)
    elif task == "both":

        y_pred[0] = np.argmax(y_pred[0], axis=1)
        y_test[0] = np.argmax(y_test[0], axis=1)

        y_pred[1] = np.argmax(y_pred[1], axis=1)
        y_test[1] = np.argmax(y_test[1], axis=1)

        con1 = confusion_matrix(y_test[0], y_pred[0])
        con2 = confusion_matrix(y_test[1], y_pred[1])
        # Accuracy plot
        plt.plot(history.history[keys[3]],label = att)
        plt.plot(history.history[keys[4]],label = att2)
        plt.title(task + " Accuracy for " + att + " and " + att2)
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend()
        plt.savefig(task + "Acc_" + att + " and " + att2 + ".png")
        plt.clf()
        # Loss plot
        plt.plot(history.history[keys[1]],label = att)
        plt.plot(history.history[keys[2]],label = att2)
        plt.title(task + " Loss for " + att + " and " + att2)
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend()
        plt.savefig(task + "Loss_" + att + " and " + att2 + ".png")
        plt.clf()
        # Confusion matrix heatmap for first attribute
        plt.imshow(con1, cmap='hot', interpolation='nearest')
        plt.title(task + " Confusion for " + att)
        plt.savefig(task + "Con_" + att + ".png")
        plt.clf()
        print(score)
        # Confusion matrix heatmap for second attribute
        plt.imshow(con2, cmap='hot', interpolation='nearest')
        plt.title(task + " Confusion for " + att2)
        plt.savefig(task + "Con_" + att2 + ".png")
        plt.clf()
        
    else:
        i = 0
        # Constructing images
        for image in y_pred:
            im = np.reshape(image,(latent_dim,10))
            im = Image.fromarray(im, mode='L')
            im.save("im[" + str(i) + "].png")
            i = i + 1