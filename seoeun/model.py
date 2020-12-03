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

#load dataset
pkl_file = open('../data/datas_new_nor.pkl', 'rb')  
datas_nor = pickle.load(pkl_file) #normalized version
pkl_file.close()

#pkl_file = open('datas.pkl', 'rb')  
#datas = pickle.load(pkl_file) #without normalization
#pkl_file.close()

def calc_test_result(result, true_label):
    preds = np.argmax(result, axis=1)
    gt = np.argmax(true_label, axis=1)
    print("Confusion Matrix :")
    print(confusion_matrix(gt, preds))
    print("Classification Report :")
    print(classification_report(gt, preds))
    print("Accuracy ", accuracy_score(gt, preds))
    tpr, fpr, _ = roc_curve(gt, 1 - result[:, 0], pos_label=0)
    print(f"AUC: {auc(fpr, tpr)}")


def build_audio_model(output_dim, use_time_distribution=True, last_activation="relu"):
    Audio_model = Sequential()
    Audio_model.add(Masking(mask_value =0,name='mask_audio'))
    Audio_model.add(Bidirectional(LSTM(20, activation='tanh', return_sequences = True, name='Bi-LSTM_audio')))
    Audio_model.add(Dropout(0.5,name='Dropout1_audio'))
    out_layer = Dense(output_dim,activation=last_activation,name='layer_2_audio')
    if use_time_distribution:
        out_layer = TimeDistributed(out_layer)
    Audio_model.add(out_layer)

    return Audio_model

def build_text_model(output_dim, use_time_distribution=True, last_activation="relu"):
    Text_model = Sequential()
    Text_model.add(Masking(mask_value =0,name='mask_text'))
    Text_model.add(Bidirectional(LSTM(125, activation='tanh', return_sequences = True, name='Bi-LSTM_text')))
    Text_model.add(Dropout(0.5,name='Dropout1_text'))
    out_layer = Dense(output_dim,activation=last_activation,name='layer_2_text')
    if use_time_distribution:
        out_layer = TimeDistributed(out_layer)
    Text_model.add(out_layer)

    return Text_model

def unimodel(datas, mode, norm,nepoch, batch_size=5):

    ################################################ (written by Seo Eun)
    # load data
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
    
    ################################################ (written by Seo Eun)
    if mode == 'audio':
        #audio
        in_audio = Input(shape=(train_audio_data.shape[1],train_audio_data.shape[2]),name='audio_input')
        Audio_model = build_audio_model(2, False, 'sigmoid')
        Audio_output = Audio_model(in_audio)
    
        model = Model(in_audio, Audio_output)
        scheduler = ExponentialDecay(initial_learning_rate=0.0003, decay_steps=(nepoch // 10) * (train_audio_data.shape[0] // batch_size), decay_rate=0.9)
        opt = Adam(learning_rate=scheduler)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(train_audio_data, train_label,
        	                epochs=nepoch,
        	                batch_size=batch_size, 
        	                shuffle=True,
                            class_weight={ 0: class_0_weight, 1: class_1_weight },
        	                validation_split=0.2)
        model.save(mode+'_'+norm+'.h5') 
        predicted_train = model.predict(train_audio_data) #709*275*2
        predicted_test = model.predict(test_audio_data) #177*275*26
        
    if mode == 'text':
        #text
        in_text = Input(shape=(train_text_data.shape[1],train_text_data.shape[2]),name='text_input')
        Text_model = build_text_model(2, False, 'sigmoid')
        Text_output = Text_model(in_text)
        model = Model(in_text, Text_output)
        scheduler = ExponentialDecay(initial_learning_rate=0.0003, decay_steps=(nepoch // 10) * (train_text_data.shape[0] // batch_size), decay_rate=0.9)
        opt = Adam(learning_rate=scheduler)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(train_text_data, train_label,
    	                    epochs=nepoch,
    	                    batch_size=batch_size,
    	                    shuffle=True,
                            class_weight={ 0: class_0_weight, 1: class_1_weight },
    	                    validation_split=0.2)
        model.save(mode+'.h5') 
        predicted_train = model.predict(train_text_data) #709*275*2
        predicted_test = model.predict(test_text_data)
    
    # predict
    print('-----train result-----')
    calc_test_result(predicted_train, train_label)
    print('-----test result-----')
    calc_test_result(predicted_test, test_label)
    
    # save model
    output = open('result_'+mode+'.pkl', 'wb')  
    pickle.dump({'pre_train':predicted_train,'pre_test':predicted_test}, output)
    output.close()     

def multimodel(datas, mode, nepoch, batch_size=5):  
    
    ################################################ (written by Seo Eun)
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

    #merging audio and text
    merged = Concatenate(axis=2)([Audio_output,Text_output])
    	 
    ################################################ (written by Dan and Seo Eun and David)
    #stage 2: BiLSTM with a two-layer neural network 
    Combined_model = Sequential() 
    Combined_model.add(Bidirectional(LSTM(150, activation='tanh', dropout=0.5, name='Bi-LSTM_merged')))
    Combined_model.add(Dropout(0.5, name='Dropout_com1'))
    #Combined_model.add(Dense(100, activation='relu', name='fc_merged1'))
    #Combined_model.add(Dropout(0.5, name='Dropout_com2'))
    Combined_model.add(Dense(2, activation='sigmoid', name='output')) 
    output = Combined_model(merged)
    #
    ################################################ (written by Seo Eun)
    model = Model([in_audio,in_text], output)
    scheduler = ExponentialDecay(initial_learning_rate=0.0003, decay_steps=(nepoch // 10) * (train_audio_data.shape[0] // batch_size), decay_rate=0.9)
    opt = Adam(learning_rate=scheduler)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit([train_audio_data, train_text_data], train_label,
    	                epochs=nepoch,
    	                batch_size=batch_size, 
    	                shuffle=True,
                        class_weight={ 0: class_0_weight, 1: class_1_weight },
    	                validation_split=0.2)
    model.save(mode+'.h5') 
    predicted_train = model.predict([train_audio_data,train_text_data]) #709*275*2
    predicted_test = model.predict([test_audio_data,test_text_data])

    print('-----train result-----')
    calc_test_result(predicted_train, train_label)
    print('-----test result-----')
    calc_test_result(predicted_test, test_label)
    
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
