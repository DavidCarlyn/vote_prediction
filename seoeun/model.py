# import all necessary packages
import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Input, BatchNormalization, LSTM, Dense, TimeDistributed, Masking, Dropout, Bidirectional, Concatenate 
from keras.models import Model
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam, Adadelta
from keras.optimizers.schedules import ExponentialDecay
from keras import backend as K 
import pickle 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

#load dataset #(written by Seo Eun)
pkl_file = open('../datas_nor.pkl', 'rb')  
datas_nor = pickle.load(pkl_file) #normalized version
pkl_file.close()

#pkl_file = open('datas.pkl', 'rb')  
#datas = pickle.load(pkl_file) #without normalization
#pkl_file.close()

# a function of presenting prediction outputs from model
# written by Seo Eun and David
def calc_test_result(result, true_label): 
    # input: (1) result: predicted one-hot coded labels
    #        (2) true_label: true one-hot coded labels
    # output: print out Confucion Matrix, Classification Report, Accuracy rate, and AUC
    preds = np.argmax(result, axis=1)
    gt = np.argmax(true_label, axis=1)
    print("Confusion Matrix :")
    print(confusion_matrix(gt, preds))
    print("Classification Report :")
    print(classification_report(gt, preds))
    print("Accuracy ", accuracy_score(gt, preds))
    tpr, fpr, _ = roc_curve(gt, 1 - result[:, 0], pos_label=0)
    print(f"AUC: {auc(fpr, tpr)}")

# a function of making a structure of bidirectional LSTM model based on audio data before compile it.
# written by David and Seo Eun
def build_audio_model(output_dim, use_time_distribution=True, last_activation="relu", return_sequences=True): 
    # input: (1) output_dim: the number of units in output layer
    #        (2) use_time_distribution: determine whether a model adds LSTM layers that return sequences rather than single values
    #        (3) last_activation: determine an activation function of the last layer
    #        (4) return_sequences: determine whether LSTM layers return sequences
    # output: (1) Audio_model: the structure of bi-LSTM model
    Audio_model = Sequential()
    Audio_model.add(Masking(mask_value =0,name='mask_audio'))
    Audio_model.add(Bidirectional(LSTM(20, activation='tanh', return_sequences = return_sequences, name='Bi-LSTM_audio')))
    Audio_model.add(Dropout(0.5,name='Dropout1_audio'))
    out_layer = Dense(output_dim,activation=last_activation,name='layer_2_audio')
    if use_time_distribution:
        out_layer = TimeDistributed(out_layer)
    Audio_model.add(out_layer)

    return Audio_model

# a function of making a structure of bidirectional LSTM model based on text data before compile it.
# written by David and Seo Eun
def build_text_model(output_dim, use_time_distribution=True, last_activation="relu", return_sequences=True):  
    # input: (1) output_dim: the number of units in output layer
    #        (2) use_time_distribution: determine whether a model adds LSTM layers that return sequences rather than single values
    #        (3) last_activation: determine an activation function of the last layer
    #        (4) return_sequences: determine whether LSTM layers return sequences
    # output: (1) Text_model: the structure of bi-LSTM model	
    Text_model = Sequential()
    Text_model.add(Masking(mask_value =0,name='mask_text'))
    Text_model.add(Bidirectional(LSTM(125, activation='tanh', return_sequences = return_sequences, name='Bi-LSTM_text')))
    Text_model.add(Dropout(0.5,name='Dropout1_text'))
    out_layer = Dense(output_dim,activation=last_activation,name='layer_2_text')
    if use_time_distribution:
        out_layer = TimeDistributed(out_layer)
    Text_model.add(out_layer)

    return Text_model

# a function of unimodal bidirectional LSTM model (contruct a model, complie, fitting, and then evaluate)
# written by Seo Eun and David 
def unimodel(datas, mode, norm,nepoch, batch_size=5):
    # input: (1) datas: training and test dataset (should be a dictionary)
    #        (2) mode: determine if unimodal is text-based ('text') or audio-based ('audio')
    #        (3) norm: determine if dataset is normalized or not
    #        (4) nepoch: the number of epochs
    #.       (5) batch_size: determine the size of batch (default=5)
    # output: print classification performances from training and test dataset, and save a trained model.
    ################################################  
    # load data 
    train_audio_data=datas['train_audio_data'].astype(np.float32)
    train_text_data=datas['train_text_data'].astype(np.float32)
    test_audio_data=datas['test_audio_data'].astype(np.float32)
    test_text_data=datas['test_text_data'].astype(np.float32)
    test_mask = datas['test_mask']
    train_mask = datas['train_mask']
    test_label = np.unique(datas['test_label'], axis=1).reshape(-1, 2)
    train_label = np.unique(datas['train_label'], axis=1).reshape(-1, 2)
    train_data = np.concatenate((train_audio_data,train_text_data), axis=2)
    test_data = np.concatenate((test_audio_data,test_text_data), axis=2)
    class_0_weight = 1.0 - ((train_label[:, 0] == 1).sum() / train_label.shape[0])
    class_1_weight = 1.0 - ((train_label[:, 1] == 1).sum() / train_label.shape[0])
    
    ################################################  
    if mode == 'audio':
        #audio
        in_audio = Input(shape=(train_audio_data.shape[1],train_audio_data.shape[2]),name='audio_input') #setting input 
        Audio_model = build_audio_model(2, False, 'sigmoid',False) #construct audio model
        Audio_output = Audio_model(in_audio) #its output
    
        model = Model(in_audio, Audio_output) #make a model
        scheduler = ExponentialDecay(initial_learning_rate=0.0003, decay_steps=(nepoch // 10) * (train_audio_data.shape[0] // batch_size), decay_rate=0.9) #learning rate with exponential decay
        opt = Adam(learning_rate=scheduler) #select optimizer
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy']) #complie it
        history = model.fit(train_audio_data, train_label, #fitting model with 0.2 validation data from training data
        	                epochs=nepoch,
        	                batch_size=batch_size, 
        	                shuffle=True,
                                class_weight={ 0: class_0_weight, 1: class_1_weight },
        	                validation_split=0.2)
        model.save(mode+'_'+norm+'.h5') #save the trained model
        predicted_train = model.predict(train_audio_data) #prediction results of training data(Dim: 709*275*2)
        predicted_test = model.predict(test_audio_data) #prediction results of test data (Dim: 177*275*26)
        
    if mode == 'text':
        #text
        in_text = Input(shape=(train_text_data.shape[1],train_text_data.shape[2]),name='text_input') #setting input 
        Text_model = build_text_model(2, False, 'sigmoid',False) #construct audio model
        Text_output = Text_model(in_text) #its output
        model = Model(in_text, Text_output) #make a model
        scheduler = ExponentialDecay(initial_learning_rate=0.0003, decay_steps=(nepoch // 10) * (train_text_data.shape[0] // batch_size), decay_rate=0.9) #learning rate with exponential decay
        opt = Adam(learning_rate=scheduler) #select optimizer
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy']) #complie it
        history = model.fit(train_text_data, train_label, #fitting model with 0.2 validation data from training data
    	                    epochs=nepoch, 
    	                    batch_size=batch_size,
    	                    shuffle=True,
                            class_weight={ 0: class_0_weight, 1: class_1_weight },
    	                    validation_split=0.2)
        model.save(mode+'.h5') #save the trained model
        predicted_train = model.predict(train_text_data) #prediction results of training data(Dim: #709*275*2)
        predicted_test = model.predict(test_text_data) #prediction results of test data (Dim: 177*275*26)
    
    # show predicted outcomes
    print('-----train result-----')
    calc_test_result(predicted_train, train_label)
    print('-----test result-----')
    calc_test_result(predicted_test, test_label)
    
    # save the final model
    output = open('result_'+mode+'.pkl', 'wb')  
    pickle.dump({'pre_train':predicted_train,'pre_test':predicted_test}, output)
    output.close()     

# a function of multimodal bidirectional LSTM model (contruct a model, complie, fitting, and then evaluate)
# written by Seo Eun, David and Dan
def multimodel(datas, mode, nepoch, batch_size=5):  
    # input: (1) datas: training and test dataset (should be a dictionary)
    #        (2) mode: determine if unimodal is text-based ('text') or audio-based ('audio') 
    #        (3) nepoch: the number of epochs
    #.       (4) batch_size: determine the size of batch (default=5)
    # output: print classification performances from training and test dataset, and save a trained model.
    
    ################################################  
    # load data
    train_audio_data=datas['train_audio_data'].astype(np.float32)
    train_text_data=datas['train_text_data'].astype(np.float32)
    test_audio_data=datas['test_audio_data'].astype(np.float32)
    test_text_data=datas['test_text_data'].astype(np.float32)
    test_mask = datas['test_mask']
    train_mask = datas['train_mask']
    test_label = np.unique(datas['test_label'], axis=1).reshape(-1, 2)
    train_label = np.unique(datas['train_label'], axis=1).reshape(-1, 2)
    train_data = np.concatenate((train_audio_data,train_text_data), axis=2)
    test_data = np.concatenate((test_audio_data,test_text_data), axis=2) 
    class_0_weight = 1.0 - ((train_label[:, 0] == 1).sum() / train_label.shape[0])
    class_1_weight = 1.0 - ((train_label[:, 1] == 1).sum() / train_label.shape[0])

    ################################################ (written by Seo Eun and David)
    #stage 1: extract unimodal features (contextual bi-LSTM) 
    #audio
    in_audio = Input(shape=(train_audio_data.shape[1],train_audio_data.shape[2]),name='audio_input')
    Audio_model = build_audio_model(1)
    Audio_output = Audio_model(in_audio) 
    #text
    in_text = Input(shape=(train_text_data.shape[1],train_text_data.shape[2]),name='text_input')
    Text_model = build_text_model(1)   
    Text_output = Text_model(in_text)

    #merging audio and text from two outputs from audio and text
    merged = Concatenate(axis=2)([Audio_output,Text_output])
    	 
    ################################################ (written by Dan, Seo Eun and David)
    #stage 2: BiLSTM with a two-layer neural network 
    Combined_model = Sequential() 
    Combined_model.add(Bidirectional(LSTM(150, activation='tanh', dropout=0.5, name='Bi-LSTM_merged'))) #bi-LSTM model
    Combined_model.add(Dropout(0.5, name='Dropout_com1')) #dropout 0.5 rate
    Combined_model.add(Dense(2, activation='sigmoid', name='output')) #output layer
    output = Combined_model(merged) 
    #
    ################################################ (written by Seo Eun)
    model = Model([in_audio,in_text], output) #model
    scheduler = ExponentialDecay(initial_learning_rate=0.0003, decay_steps=(nepoch // 10) * (train_audio_data.shape[0] // batch_size), decay_rate=0.9) #learning rate with exponential Decay
    opt = Adam(learning_rate=scheduler) #use Adam optimizer
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy']) #compile a model
    history = model.fit([train_audio_data, train_text_data], train_label, #fitting a model
    	                epochs=nepoch,
    	                batch_size=batch_size, 
    	                shuffle=True,
                        class_weight={ 0: class_0_weight, 1: class_1_weight },
    	                validation_split=0.2)
    model.save(mode+'.h5') #save the trained model
    predicted_train = model.predict([train_audio_data,train_text_data]) #prediction results of training data(Dim: #709*2)
    predicted_test = model.predict([test_audio_data,test_text_data]) #prediction results of test data(Dim: #177*2)

    # show predicted outcomes
    print('-----train result-----')
    calc_test_result(predicted_train, train_label)
    print('-----test result-----')
    calc_test_result(predicted_test, test_label)

    # scatter plot of results (written by Dan Weber)
    plt.scatter(range(180), predicted_test, c='r')
    plt.scatter(range(180), test_label, c='g')
    plt.title('Prediction Accuracy')
    plt.ylabel('test case label')
    plt.xlabel('test case data')
    plt.show()

    # Training and validation loss plot (Written by Dan Weber)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()    
    
    # save the trained model
    output = open('result_'+mode+'.pkl', 'wb')  
    pickle.dump({'pre_train':predicted_train,'pre_test':predicted_test}, output)
    output.close() 
    
if __name__=="__main__":

    print('----- multimodal -----')
    #multimodel(datas,'multimodal_no_normalized',50)
    multimodel(datas_nor,'multimodal_normalized', 50, 5)
    print('----- text -----')
    #unimodel(datas,'text','no_normalized',50)
    unimodel(datas_nor,'text','normalized',50, 5)
    print('----- audio -----')
    #unimodel(datas,'audio','no_normalized',50)
    unimodel(datas_nor,'audio','normalized',50, 5)
