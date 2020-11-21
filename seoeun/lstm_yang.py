import numpy as np
import pandas as pd
from keras.layers import Input, LSTM, Dense, TimeDistributed, Masking, Dropout, Bidirectional, Concatenate
from keras.models import Model
from keras import backend as K
import theano.tensor as T
import theano
import pickle
import sys
import argparse
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
import os

np.random.seed(1337) # for reproducibility
path = "/Volumes/Backup Plus/Audio_data/feature"
os.chdir(path)

#load final feature matrix
pkl_file = open('final_features_forLSTM.pkl', 'rb')  
feature = pickle.load(pkl_file)
pkl_file.close() 

#extract case id
ids = list(feature.keys())

#save labels (DV) and calculate sentences length
labels = []
sentence_len = [] 
for i in range(len(ids)):
    labels.append(feature[ids[i]]['DV'])
    sentence_len.append(feature[ids[i]]['feature'].shape[0])
    
# maximum number of sentences    
maxlen = max(sentence_len)

# location of audio and text features
name_col=[]
for j in range(26):
    name_col.append(j)
name_col2=[]    
for j in range(26,226):
    name_col2.append(j)

# create audio matrix and text matrix for contextual bi-LSTM
#data_X = [] 
audio_data_X = []
text_data_X = [] 
sen_length = []
for i in range(len(ids)):
    all_data = feature[ids[i]]['feature'] 
    n = all_data.shape[0]
    m = all_data.shape[1]
    add = np.array([0]*m)  
    for j in range(maxlen-n):
        all_data = np.row_stack((all_data,add))  
    audio_data = np.asarray(pd.DataFrame(all_data)[name_col])
    text_data = np.asarray(pd.DataFrame(all_data)[name_col2])
    #data_X.append(all_data)
    audio_data_X.append(audio_data)
    text_data_X.append(text_data)
    sen_length.append(n) 
     
audio_data_X=np.asarray(audio_data_X) #886, 275, 26
text_data_X=np.asarray(text_data_X) #886, 275, 200
#data_X=np.asarray(data_X) #886, 275, 226

# create mask matrix and createOneHot label
mask_data = np.zeros((len(ids),maxlen), dtype='float')
label_Y = np.zeros((len(ids), 2))
for i in range(len(ids)):
    mask_data[i,:sen_length[i]]=1.0 
    label_Y[i,labels[i]]=1
    
label_Y = label_Y.astype('int')  #change to integer  
   
# create test, train, and validation matrix
test_portion = 0.2 
n_samples = text_data_X.shape[0];
sidx = np.arange(n_samples) ;
np.random.shuffle(sidx);
n_test = int(np.round(n_samples * test_portion))  

test_ids = np.array(ids)[sidx[:n_test]] #ids used for test dataset
train_ids = np.array(ids)[sidx[n_test:]] #ids used for trin dataset

test_audio_data = np.asarray([audio_data_X[s] for s in sidx[:n_test]])
test_text_data = np.asarray([text_data_X[s] for s in sidx[:n_test]])
test_mask = np.asarray([mask_data[s] for s in sidx[:n_test]])
test_label = np.asarray([label_Y[s] for s in sidx[:n_test]])

train_audio_data = np.asarray([audio_data_X[s] for s in sidx[n_test:]])
train_text_data = np.asarray([text_data_X[s] for s in sidx[n_test:]])
train_mask = np.asarray([mask_data[s] for s in sidx[n_test:]])
train_label = np.asarray([label_Y[s] for s in sidx[n_test:]])
 
datas = {'test_audio_data':test_audio_data,'test_text_data':test_text_data,
        'test_mask':test_mask,'test_label':test_label,
        'train_audio_data':train_audio_data,'train_text_data':train_text_data,
        'train_mask':train_mask,'train_label':train_label}

#Evaluate results
def calc_test_result(result, test_label, test_mask):

	true_label=[]
	predicted_label=[]

	for i in range(result.shape[0]):
		for j in range(result.shape[1]):
			if test_mask[i,j]==1:
				true_label.append(np.argmax(test_label[i,j] ))
				predicted_label.append(np.argmax(result[i,j] ))
		
	print("Confusion Matrix :")
	print(confusion_matrix(true_label, predicted_label))
	print("Classification Report :")
	print(classification_report(true_label, predicted_label))
	print("Accuracy ", accuracy_score(true_label, predicted_label))
  
    
def model(datas):  
    
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
    input_data = Input(shape=(train_data.shape[1],train_data.shape[2]))
	masked = Masking(mask_value =0)(input_data)
	lstm = Bidirectional(LSTM(300, activation='tanh', return_sequences = True, dropout=0.4))(masked)
	inter = Dropout(0.9)(lstm)
	inter1 = TimeDistributed(Dense(500,activation='relu'))(inter)
	inter = Dropout(0.9)(inter1)
	merged = TimeDistributed(Dense(2,activation='linear'))(inter)
    
    ################################################ (written by Dan)
    #stage 2: BiLSTM with a two-layer neural network


    ################################################
    # compile
    # below is example. Need to be revised
	model = Model(input_data, output)  
	model.compile(optimizer='adadelta', loss='categorical_crossentropy', sample_weight_mode='temporal')
	early_stopping = EarlyStopping(monitor='val_loss', patience=10)
	model.fit(train_data, train_label,
	                epochs=200,
	                batch_size=10,
	                sample_weight=train_mask,
	                shuffle=True, 
	                callbacks=[early_stopping],
	                validation_split=0.2)
	                
	model.save('./models/'+mode+'.h5') 

	train_activations = aux.predict(train_data)
	test_activations = aux.predict(test_data)
	 
    result = model.predict(test_data)
	calc_test_result(result, test_label, test_mask)
 
    
    
if __name__=="__main__":
	
	unimodal(mode)
 
