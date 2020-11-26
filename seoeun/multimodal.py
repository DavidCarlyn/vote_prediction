import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Input, BatchNormalization, LSTM, Dense, TimeDistributed, Masking, Dropout, Bidirectional, Concatenate 
from keras.models import Model
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam, Adadelta
from keras import backend as K 
import pickle 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score 
from keras.callbacks import EarlyStopping

#load dataset
pkl_file = open('datas_new_nor.pkl', 'rb')  
datas_nor = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('datas_new.pkl', 'rb')  
datas = pickle.load(pkl_file)
pkl_file.close()

def calc_test_result(result, true_label):
	tr_label=[]
	predicted_label=[]
    for i in range(result.shape[0]): 
    	tr_label.append(np.argmax(true_label[i]))
    	predicted_label.append(np.argmax(result[i]))
	print("Confusion Matrix :")
	print(confusion_matrix(tr_label, predicted_label))
	print("Classification Report :")
	print(classification_report(tr_label, predicted_label))
	print("Accuracy ", accuracy_score(tr_label, predicted_label))


def unimodel(datas,mode,norm,nepoch):

    ################################################ (written by Seo Eun)
    # load data
    train_audio_data=datas['train_audio_data'] 
    train_text_data=datas['train_text_data']
    test_audio_data=datas['test_audio_data'] 
    test_text_data=datas['test_text_data'] 
    test_mask = datas['test_mask']
    train_mask = datas['train_mask']
    test_label = datas['test_label']
    train_label = datas['train_label'] 
    train_data = np.concatenate((train_audio_data,train_text_data), axis=2)
    test_data = np.concatenate((test_audio_data,test_text_data), axis=2)
    
    ################################################ (written by Seo Eun)
    if mode == 'audio':
        #audio
        in_audio = Input(shape=(train_audio_data.shape[1],train_audio_data.shape[2]),name='audio_input')
        Audio_model = Sequential()
        Audio_model.add(Masking(mask_value =0,name='mask_audio'))
        Audio_model.add(Bidirectional(LSTM(300, activation='tanh', return_sequences = True, dropout=0.5, name='Bi-LSTM_audio')))
        Audio_model.add(Dropout(0.5,name='Dropout1_audio'))
        Audio_model.add(TimeDistributed(Dense(500,activation='relu',name='TimeDistributed1_audio')))
        Audio_model.add(Dropout(0.5,name='Dropout2_audio'))
        Audio_model.add(TimeDistributed(Dense(1,activation='relu',name='TimeDistributed2_audio')))
        Audio_model.add(Dropout(0.5,name='Dropout3_audio'))
        Audio_model.add(Dense(2, activation='sigmoid', name='output')) 
        Audio_output = Audio_model(in_audio)
    
        model = Model(in_audio, Audio_output)
        model.compile(optimizer='adam', loss='binary_crossentropy', sample_weight_mode='temporal',metrics=['accuracy'])
        history = model.fit(train_audio_data, train_label,
        	                epochs=nepoch,
        	                batch_size=35,
        	                sample_weight = train_mask,
        	                shuffle=True, 
        	                callbacks=[EarlyStopping(monitor='val_loss', patience=10)],
        	                validation_split=0.2)
        model.save(mode+'_'+norm+'.h5') 
        predicted_train = model.predict(train_audio_data) #709*275*2
        predicted_test = model.predict(test_audio_data) #177*275*26
        
    if mode == 'text':
        #text
        in_text = Input(shape=(train_text_data.shape[1],train_text_data.shape[2]),name='text_input')
        Text_model = Sequential()
        Text_model.add(Masking(mask_value =0,name='mask_text'))
        Text_model.add(Bidirectional(LSTM(300, activation='tanh', return_sequences = True, dropout=0.5, name='Bi-LSTM_text')))
        Text_model.add(Dropout(0.5,name='Dropout1_text'))
        Text_model.add(TimeDistributed(Dense(500,activation='relu',name='TimeDistributed1_text')))
        Text_model.add(Dropout(0.5,name='Dropout2_text'))
        Text_model.add(TimeDistributed(Dense(1,activation='relu',name='TimeDistributed2_text')))
        Text_model.add(Dropout(0.5,name='Dropout3_text'))   
        Text_model.add(Dense(2, activation='sigmoid', name='output')) 
        Text_output = Text_model(in_text)
  
        model = Model(in_text, Text_output)
        model.compile(optimizer='adam', loss='binary_crossentropy', sample_weight_mode='temporal',metrics=['accuracy'])
        history = model.fit(train_text_data, train_label,
        	                epochs=nepoch,
        	                batch_size=35,
        	                sample_weight = train_mask,
        	                shuffle=True, 
        	                callbacks=[EarlyStopping(monitor='val_loss', patience=10)],
        	                validation_split=0.2)
        model.save(mode+'.h5') 
        predicted_train = model.predict(train_text_data) #709*275*2
        predicted_test = model.predict(test_text_data)
    
    #predict
    print('-----train result-----')
    calc_test_result(np.mean(predicted_train,axis=1), np.mean(train_label,axis=1,dtype='int'))
    print('-----test result-----')
    calc_test_result(np.mean(predicted_test,axis=1), np.mean(test_label,axis=1,dtype='int'))
    #
    output = open('result_'+mode+'.pkl', 'wb')  
    pickle.dump({'pre_train':predicted_train,'pre_test':predicted_test}, output)
    output.close()     

def multimodel(datas,mode,nepoch):  
    
    ################################################ (written by Seo Eun)
    # load data
    train_audio_data=datas['train_audio_data'] 
    train_text_data=datas['train_text_data']
    test_audio_data=datas['test_audio_data'] 
    test_text_data=datas['test_text_data'] 
    test_mask = datas['test_mask']
    train_mask = datas['train_mask']
    test_label = datas['test_label']
    train_label = datas['train_label'] 
    train_data = np.concatenate((train_audio_data,train_text_data), axis=2)
    test_data = np.concatenate((test_audio_data,test_text_data), axis=2)
    
    ################################################ (written by Seo Eun)
    #stage 1: extract unimodal features (contextual bi-LSTM) 
    #audio
    in_audio = Input(shape=(train_audio_data.shape[1],train_audio_data.shape[2]),name='audio_input')
    Audio_model = Sequential()
    Audio_model.add(Masking(mask_value =0,name='mask_audio'))
    Audio_model.add(Bidirectional(LSTM(300, activation='tanh', return_sequences = True, dropout=0.5, name='Bi-LSTM_audio')))
    Audio_model.add(Dropout(0.5,name='Dropout1_audio'))
    Audio_model.add(TimeDistributed(Dense(500,activation='relu',name='TimeDistributed1_audio')))
    Audio_model.add(Dropout(0.5,name='Dropout2_audio'))
    Audio_model.add(TimeDistributed(Dense(1,activation='relu',name='TimeDistributed2_audio')))
    Audio_model.add(Dropout(0.5,name='Dropout3_audio'))
    Audio_output = Audio_model(in_audio)
    #text
    in_text = Input(shape=(train_text_data.shape[1],train_text_data.shape[2]),name='text_input')
    Text_model = Sequential()
    Text_model.add(Masking(mask_value =0,name='mask_text'))
    Text_model.add(Bidirectional(LSTM(300, activation='tanh', return_sequences = True, dropout=0.5, name='Bi-LSTM_text')))
    Text_model.add(Dropout(0.5,name='Dropout1_text'))
    Text_model.add(TimeDistributed(Dense(500,activation='relu',name='TimeDistributed1_text')))
    Text_model.add(Dropout(0.5,name='Dropout2_text'))
    Text_model.add(TimeDistributed(Dense(1,activation='relu',name='TimeDistributed2_text')))
    Text_model.add(Dropout(0.5,name='Dropout3_text'))   
    Text_output = Text_model(in_text)
    #merging audio and text
    merged = Concatenate(axis=2)([Audio_output,Text_output])
    	 
    ################################################ (written by Dan and Seo Eun)
    #stage 2: BiLSTM with a two-layer neural network 
    Combined_model = Sequential() 
    Combined_model.add(Bidirectional(LSTM(100, activation='tanh', return_sequences = True, dropout=0.5, name='Bi-LSTM_merged')))
    Combined_model.add(Dropout(0.5, name='Dropout_com1'))
    Combined_model.add(Dense(30, activation='relu', name='fc_merged'))
    Combined_model.add(Dropout(0.5, name='Dropout_com2'))
    Combined_model.add(Dense(2, activation='sigmoid', name='output')) 
    output = Combined_model(merged)
    #
    ################################################ (written by Seo Eun)
    model = Model([in_audio,in_text], output)
    model.compile(optimizer='adam', loss='binary_crossentropy', sample_weight_mode='temporal',metrics=['accuracy'])
    history = model.fit([train_audio_data,train_text_data], train_label,
    	                epochs=nepoch,
    	                batch_size=35,
    	                sample_weight = train_mask,
    	                shuffle=True, 
    	                callbacks=[EarlyStopping(monitor='val_loss', patience=10)],
    	                validation_split=0.2)
    model.save(mode+'.h5') 
    predicted_train = model.predict([train_audio_data,train_text_data]) #709*275*2
    predicted_test = model.predict([test_audio_data,test_text_data])
    #
    print('-----train result-----')
    calc_test_result(np.mean(predicted_train,axis=1), np.mean(train_label,axis=1,dtype='int'))
    print('-----test result-----')
    calc_test_result(np.mean(predicted_test,axis=1), np.mean(test_label,axis=1,dtype='int'))
    #
    output = open('result_'+mode+'.pkl', 'wb')  
    pickle.dump({'pre_train':predicted_train,'pre_test':predicted_test}, output)
    output.close() 
    
if __name__=="__main__":
	
    print('----- multimodal -----')
	multimodel(datas,'multimodal_no_normalized',50)
    multimodel(datas_nor,'multimodal_normalized',50)
    print('----- text -----')
    unimodel(datas,'text','no_normalized',50)
    unimodel(datas_nor,'text','normalized',50)
    print('----- audio -----')
    unimodel(datas,'audio','no_normalized',50)
    unimodel(datas_nor,'audio','normalized',50)
