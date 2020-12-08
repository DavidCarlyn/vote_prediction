To run SVM Experiments:
    python svm_classifier.py

To run NN Experiemnts:
    Both: 
        python naive_nn.py --bias --shuffle --exp both
    Audio:
        python naive_nn.py --bias --shuffle --exp audio
    Textual:
        python naive_nn.py --bias --shuffle --exp text

To run BiLSTM Experiments:
    python model.py
    
    note: datas.pkl and datas_nor.pkl must also be in the same directory as model.py
