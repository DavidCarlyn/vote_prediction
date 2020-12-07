import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

# random.seed(5523)

# Written by Seo Eun Yang lines 14-54

pkl_file = open('data.nosync/datas_new.pkl', 'rb')
datas = pickle.load(pkl_file)
pkl_file.close()

train_audio_data = datas['train_audio_data']  # 709*275*26
train_text_data = datas['train_text_data']  # 709*275*200
test_audio_data = datas['test_audio_data']  # 177*275*26
test_text_data = datas['test_text_data']  # 177*275*200
train_mask = datas['train_mask']  # 709*275 show which sentences are real ones
test_mask = datas['test_mask']  # 177*275 show which sentences are real ones

# Since each judicial case has different number of sentences, zero-padding has been applied for Deep NN.
# Thus, to get back the original dataset which has distinct number of sentences,
# I first take a sum of each judicial case across sentences (axis=1)
# then divided by the genuine number of sentences in each case

# take a sum of judicial case across sentences
sum_train_audio = np.sum(train_audio_data, axis=1)  # 709*26
sum_train_text = np.sum(train_text_data, axis=1)  # 709*200
sum_test_audio = np.sum(test_audio_data, axis=1)  # 177*26
sum_test_text = np.sum(test_text_data, axis=1)  # 177*200

# the genuine number of sentences in each judicial case
sentence_len_train = np.sum(train_mask, axis=1)  # 709
sentence_len_test = np.sum(test_mask, axis=1)  # 177

# calculte the mean of each case.
mean_train_audio, mean_train_text, mean_test_audio, mean_test_text = [], [], [], []
for i in range(len(sum_train_audio)):
    mean_train_audio.append(sum_train_audio[i] / sentence_len_train[i])
    mean_train_text.append(sum_train_text[i] / sentence_len_train[i])

for i in range(len(sum_test_audio)):
    mean_test_audio.append(sum_test_audio[i] / sentence_len_train[i])
    mean_test_text.append(sum_test_text[i] / sentence_len_train[i])

mean_train_audio = np.array(mean_train_audio).astype(float)  # 709*26
mean_train_text = np.array(mean_train_text).astype(float)  # 709*200
mean_test_audio = np.array(mean_test_audio).astype(float)  # 177*26
mean_test_text = np.array(mean_test_text).astype(float)  # 177*200

# Written by Michael Cooch lines 58-210
# create dataframes for multimodal model
X_train = np.append(mean_train_audio, mean_train_text, axis=1).astype(float)
X_test = np.append(mean_test_audio, mean_test_text, axis=1).astype(float)

# extract labels for testing and train set
train_label = []
for x in datas['train_label']:
    train_label.append(x[0][1])

test_label = []
for x in datas['test_label']:
    test_label.append(x[0][1])

y_train = np.array(train_label)
y_test = np.array(test_label)

# np.savetxt('X_train.csv', X_train, delimiter=",")
# np.savetxt('X_test.csv', X_test, delimiter=",")
# np.savetxt('y_train.csv', y_train, delimiter=",")
# np.savetxt('y_test.csv', y_test, delimiter=",")


# Written by Michael Cooch

def plot_roc(fpr_linear, tpr_linear, fpr_rbf, tpr_rbf, fpr_sigmoid, tpr_sigmoid, auc_linear, auc_rbf, auc_sigmoid, title):
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_linear, tpr_linear, label='Linear SVM (area = %0.2f)' % auc_linear)
    plt.plot(fpr_rbf, tpr_rbf, label='SVM with RBF kernel (area = %0.2f)' % auc_rbf)
    plt.plot(fpr_sigmoid, tpr_sigmoid, label='SVM with sigmoid kernel (area = %0.2f)' % auc_sigmoid)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curve ' + title)
    plt.legend(loc='best')
    plt.savefig('roc_plot_' + title)


def svm_linear(X_train, X_test, y_train, y_test):

    classifier = svm.SVC(kernel='linear', probability=True)
    linear_svm_fit = classifier.fit(X_train, y_train)
    print("Liner SVM Train Accuracy ", linear_svm_fit.score(X_train, y_train))
    # get predictions for the test set
    y_pred_linear = classifier.predict(X_test)
    print("Linear SVM Test Accuracy: ", metrics.accuracy_score(y_test, y_pred_linear))
    # get confusion matrix/metrics
    print(metrics.confusion_matrix(y_test, y_pred_linear))
    print(metrics.classification_report(y_test, y_pred_linear))
    # get metrics to build roc plot
    y_pred_probs_linear = linear_svm_fit.predict_proba(X_test)[:, 1]
    fpr_linear, tpr_linear, thresholds_linear = metrics.roc_curve(y_test, y_pred_probs_linear)
    auc_linear = metrics.auc(fpr_linear, tpr_linear)

    return fpr_linear, tpr_linear, auc_linear


def svm_rbf(X_train, X_test, y_train, y_test):

    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                  'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
    # performs cross validation to fit the SVC model
    grid_search_rbf = GridSearchCV(svm.SVC(kernel='rbf', probability=True), param_grid=param_grid,
                                   scoring='accuracy', cv=5)
    rbf_fit = grid_search_rbf.fit(X_train, y_train)
    print("SVM RBF Kernel Train Accuracy: ", rbf_fit.score(X_train, y_train))
    # get predictions for the test set
    preds_rbf_grid = rbf_fit.predict(X_test)
    print("SVM RBF Kernel Test Accuracy: ", metrics.accuracy_score(y_test, preds_rbf_grid))
    # get confusion matrix/metrics, best parameters from cross validation
    print(metrics.confusion_matrix(y_test, preds_rbf_grid))
    print(metrics.classification_report(y_test, preds_rbf_grid))
    print(rbf_fit.best_params_)
    # get metrics to build roc plot
    y_pred_probs_rbf = rbf_fit.predict_proba(X_test)[:, 1]
    fpr_rbf, tpr_rbf, thresholds_rbf = metrics.roc_curve(y_test, y_pred_probs_rbf)
    auc_rbf = metrics.auc(fpr_rbf, tpr_rbf)

    return fpr_rbf, tpr_rbf, auc_rbf


def svm_sigmoid(X_train, X_test, y_train, y_test):

    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                  'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
    # performs cross validation to fit the SVC model
    grid_search_sigmoid = GridSearchCV(svm.SVC(kernel='sigmoid', probability=True), param_grid=param_grid,
                                       scoring='accuracy', cv=5)
    sigmoid_fit = grid_search_sigmoid.fit(X_train, y_train)
    print("SVM Sigmoid Kernel Train Accuracy: ", sigmoid_fit.score(X_train, y_train))
    # get predictions for the test set
    preds_sigmoid_grid = sigmoid_fit.predict(X_test)
    print("SVM Sigmoid Kernel Test Accuracy: ", metrics.accuracy_score(y_test, preds_sigmoid_grid))
    # get confusion matrix/metrics, best parameters from cross validation
    print(metrics.confusion_matrix(y_test, preds_sigmoid_grid))
    print(metrics.classification_report(y_test, preds_sigmoid_grid))
    print(sigmoid_fit.best_params_)
    # get metrics to build roc plot
    y_pred_probs_sigmoid = sigmoid_fit.predict_proba(X_test)[:, 1]
    fpr_sigmoid, tpr_sigmoid, rbf = metrics.roc_curve(y_test, y_pred_probs_sigmoid)
    auc_sigmoid = metrics.auc(fpr_sigmoid, tpr_sigmoid)

    return fpr_sigmoid, tpr_sigmoid, auc_sigmoid


def unimodel(mean_train_audio, mean_train_text, mean_test_audio, mean_test_text, y_train, y_test, mode):
    # scale audio data
    scaler_audio = StandardScaler()
    scaler_audio.fit(mean_train_audio)
    mean_train_audio = scaler_audio.transform(mean_train_audio)
    mean_test_audio = scaler_audio.transform(mean_test_audio)

    # scale text audio
    scaler_text = StandardScaler()
    scaler_text.fit(mean_train_text)
    mean_train_text = scaler_text.transform(mean_train_text)
    mean_test_text = scaler_text.transform(mean_test_text)

    if mode == 'audio':
        # audio
        # Linear SVM
        fpr_linear, tpr_linear, auc_linear = svm_linear(mean_train_audio, mean_test_audio, y_train, y_test)
        # SVM with RBF Kernel
        fpr_rbf, tpr_rbf, auc_rbf = svm_rbf(mean_train_audio, mean_test_audio, y_train, y_test)
        # SVM with Sigmoid Kernel
        fpr_sigmoid, tpr_sigmoid, auc_sigmoid = svm_sigmoid(mean_train_audio, mean_test_audio, y_train, y_test)
        # plot audio ROC
        plot_roc(fpr_linear, tpr_linear, fpr_rbf, tpr_rbf, fpr_sigmoid, tpr_sigmoid, auc_linear, auc_rbf, auc_sigmoid, 'audio')
    elif mode == 'text':
        # text
        # Linear SVM
        fpr_linear, tpr_linear, auc_linear = svm_linear(mean_train_text, mean_test_text, y_train, y_test)
        # SVM with RBF Kernel
        fpr_rbf, tpr_rbf, auc_rbf = svm_rbf(mean_train_text, mean_test_text, y_train, y_test)
        # SVM with Sigmoid Kernel
        fpr_sigmoid, tpr_sigmoid, auc_sigmoid = svm_sigmoid(mean_train_text, mean_test_text, y_train, y_test)
        # plot text ROC
        plot_roc(fpr_linear, tpr_linear, fpr_rbf, tpr_rbf, fpr_sigmoid, tpr_sigmoid, auc_linear, auc_rbf, auc_sigmoid, 'text')


def multimodel(X_train, X_test, y_train, y_test):
    # scale combined data
    scaler_audio = StandardScaler()
    scaler_audio.fit(X_train)
    X_train = scaler_audio.transform(X_train)
    X_test = scaler_audio.transform(X_test)

    # Linear SVM
    fpr_linear, tpr_linear, auc_linear = svm_linear(X_train, X_test, y_train, y_test)
    # SVM with RBF Kernel
    fpr_rbf, tpr_rbf, auc_rbf = svm_rbf(X_train, X_test, y_train, y_test)
    # SVM with Sigmoid Kernel
    fpr_sigmoid, tpr_sigmoid, auc_sigmoid = svm_sigmoid(X_train, X_test, y_train, y_test)
    # plot multimodal ROC
    plot_roc(fpr_linear, tpr_linear, fpr_rbf, tpr_rbf, fpr_sigmoid, tpr_sigmoid, auc_linear, auc_rbf, auc_sigmoid, 'audio + text')


if __name__ == "__main__":
    # multimodal SVC
    print('----- multimodal -----')
    multimodel(X_train, X_test, y_train, y_test)
    # unimodal text SVC
    print('----- text -----')
    unimodel(mean_train_audio, mean_train_text, mean_test_audio, mean_test_text, y_train, y_test, 'text')
    # unimodal audio SVC
    print('----- audio -----')
    unimodel(mean_train_audio, mean_train_text, mean_test_audio, mean_test_text, y_train, y_test, 'audio')
