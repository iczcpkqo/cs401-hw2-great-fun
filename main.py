import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import time
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import joblib
import numba as nb
import sys

# MYPATH = './drive/MyDrive/codedomain/workspace/modules/cs401_machine-learning_neural-networks/homework/hw2/cs401-hw2-great-fun'
MYPATH = '.'

TRAIN_DATA = MYPATH + "/train_data/"
RESULT_DATA_PATH = MYPATH + "/result_data/"
RESULT_REPORT_PATH = MYPATH + "/result_report/"
RESULT_GRAPH_PATH = MYPATH + "/result_graph/"
MODEL_PATH = MYPATH + "/trained_model/"
PRESENT_TIME = str(time.strftime("%Y-%m-%d %H.%M.%S", time.localtime(time.time())))


def run():
    # SETTING AND ON-OFF
    args = {'is_train': True,
            'is_print_report': True,
            'is_print_pred': True,
            'is_save_model': True,
            'is_probability': True,
            'is_graph_roc': True,
            'is_graph_auc': False,
            'C': 0.75,
            'test_size': 0.2,
            'pause_time': 0.00,
            'kernel': 'Polynomial',
            'model_name': '0.5_SV_Sigmoid__2020-12-18 05.29.13.m',
            'train_data': 'train-io-tiny.txt',
            'test_data': 'test-in-tiny.txt',
            'train_col': ['D-1', 'D-2', 'D-3', 'D-4', 'D-5', 'D-6', 'D-7', 'D-8', 'D-9', 'D-10', 'D-11', 'D-12', 'class'],
            'test_col': ['D-1', 'D-2', 'D-3', 'D-4', 'D-5', 'D-6', 'D-7', 'D-8', 'D-9', 'D-10', 'D-11', 'D-12']}

    # get parameter
    c = args['C']
    is_train = args['is_train']

    # for big_C in np.arange(0.1, 1.05, 0.1):
    print('laod the C='+str(c))

    if is_train:
        y_test, y_test_probability, y_pred, test_pred = run_train(args)
    elif not is_train:
        y_test, y_test_probability, y_pred, test_pred = run_model(args)

    # Plotting ROC
    # draw_roc(args, y_test=y_test, y_test_probability=y_test_probability)
    print('start draw graph')
    show_graph(args, y_test=y_test, y_test_probability=y_test_probability)

    # Save data
    print('save data')
    save_data(args, y_test=y_test, y_pred=y_pred, test_pred=test_pred)

# for i in range(2, 3):
    #     run_case(i,big_C)

    # c = args['C']
    # kernel = args['kernel']
    # is_report = args['is_print_report']
    # is_pred = args['is_print_pred']
    # y_test = kwargs['y_test']
    # y_pred = kwargs['y_pred']
    # test_pred = kwargs['test_pred']



def run_train(args):
    c = args['C']
    te_co = args['test_col']
    te_da = args['test_data']
    te_si = args['test_size']
    tr_co = args['train_col']
    tr_da = args['train_data']
    kernel = args['kernel']
    proba = args['is_probability']

    # Data
    ## data
    url = TRAIN_DATA + tr_da
    pre_url = TRAIN_DATA + te_da

    # Assign colum names to the dataset
    colnames = tr_co
    pre_colnames = te_co

    # Read dataset to pandas dataframe
    print('read data file')
    train_data = pd.read_csv(url, names=colnames, sep=' ')
    pre_data = pd.read_csv(pre_url, names=pre_colnames, sep=' ')

    # Pre-processing
    print('drop data column')
    X = train_data.drop(tr_co[-1], axis=1)
    y = train_data[tr_co[-1]]

    # Separate data
    print('split data')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=te_si)

    # Select classifier
    ## Gauss
    print('select kernel')
    if kernel == 'Gaussian':
        print(kernel)
        svclassifier = SVC(kernel='rbf', probability=proba, C=c)

    ## Polynomial
    if kernel == 'Polynomial':
        print(kernel)
        svclassifier = SVC(kernel='poly', probability=proba, degree=8, C=c)
        svclassifier.fit(X_train, y_train)

    ## sigmoid
    if kernel == 'Sigmoid':
        print(kernel)
        svclassifier = SVC(kernel='sigmoid', probability=proba, C=c)

    # Training
    print('fit train data')
    svclassifier.fit(X_train, y_train)

    # Save model
    print('save model')
    joblib.dump(svclassifier, MODEL_PATH + str(c) + '_SV_' + kernel + '__' + PRESENT_TIME + '.m')

    # Plotting ROC
    ## Score obtained
    print('get probability')
    y_test_probability = svclassifier.decision_function(X_test)
    # print(y_test_probability)

    ## Obtain true/false rate
    # fpr, tpr, threshold = roc_curve(y_test, y_test_probability)
    # print(fpr)
    # print(tpr)
    # print(threshold)

    # Predictive Assessment
    print('predict y_pred')
    y_pred = svclassifier.predict(X_test)
    print('predict test_pred')
    test_pred = svclassifier.predict(pre_data)

    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)
    # pd.set_option('max_colwidth', 10000000)
    np.set_printoptions(threshold=sys.maxsize)


    # print(fpr)
    # print(tpr)
    # print(threshold)

    # fpr, tpr, threshold = roc_curve(y_test, y_pred)
    # print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test, y_pred))
    # print(y_pred)

    # name_report_file = ''
    # name__file= ''
    # if is_sel == 0:
    # ======  ======  ======  ======  ======  ======  ======
    # file = open(RESULT_REPORT_PATH + str(c) + '_' + kernel + '_Report__' + PRESENT_TIME + '.txt', 'w') file.write(str(confusion_matrix(y_test, y_pred)))
    # file.write('\n')
    # file.write(str(classification_report(y_test, y_pred)))
    # file.write('\n')
    # file.write(str(y_pred))
    # file.close()
    #
    # file = open(RESULT_DATA_PATH + str(c) + '_' + kernel + '_Out__' + PRESENT_TIME + '.txt', 'w')
    # file.write(str(test_pred))
    # file.close()


    return y_test, y_test_probability, y_pred, test_pred


def run_model(args):
    is_roc = args['is_graph_roc']
    te_co = args['test_col']
    te_da = args['test_data']
    te_si = args['test_size']
    tr_co = args['train_col']
    tr_da = args['train_data']
    kernel = args['kernel']
    proba = args['is_probability']
    model_name = args['model_name']

    # Data
    ## data
    url = TRAIN_DATA + tr_da
    pre_url = TRAIN_DATA + te_da

    # Assign colum names to the dataset
    colnames = tr_co
    pre_colnames = te_co

    # Read dataset to pandas dataframe
    print('read data file')
    train_data = pd.read_csv(url, names=colnames, sep=' ')
    pre_data = pd.read_csv(pre_url, names=pre_colnames, sep=' ')

    # Pre-processing
    print('drop data column')
    X = train_data.drop(tr_co[-1], axis=1)
    y = train_data[tr_co[-1]]

    # Separate data
    print('split data')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=te_si)

    # Calling the model
    print('start joblib')
    svclassifier = joblib.load(MODEL_PATH + model_name)

    # Plotting ROC
    ## Score obtained
    y_test_probability = None
    if is_roc:
        print('take probability')
        y_test_probability = svclassifier.decision_function(X_test)
    # print(y_test_probability)

    ## Obtain true/false rate
    # fpr, tpr, threshold = roc_curve(y_test, y_test_probability)
    # print(fpr)
    # print(tpr)
    # print(threshold)

    # Projections
    print('predict y_pred & test_pred')
    y_pred = svclassifier.predict(X_test)
    print('predict test_pred')
    test_pred = svclassifier.predict(pre_data)

    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)
    # pd.set_option('max_colwidth', 10000000)
    np.set_printoptions(threshold=sys.maxsize)


    # print(fpr)
    # print(tpr)
    # print(threshold)

    # fpr, tpr, threshold = roc_curve(y_test, y_pred)
    # print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test, y_pred))
    # print(y_pred)

    # name_report_file = ''
    # name__file= ''
    # if is_sel == 0:
    # ======  ======  ======  ======  ======  ======  ======
    # file = open(RESULT_REPORT_PATH + str(c) + '_' + kernel + '_Report__' + PRESENT_TIME + '.txt', 'w') file.write(str(confusion_matrix(y_test, y_pred)))
    # file.write('\n')
    # file.write(str(classification_report(y_test, y_pred)))
    # file.write('\n')
    # file.write(str(y_pred))
    # file.close()
    #
    # file = open(RESULT_DATA_PATH + str(c) + '_' + kernel + '_Out__' + PRESENT_TIME + '.txt', 'w')
    # file.write(str(test_pred))
    # file.close()


    return y_test, y_test_probability, y_pred, test_pred


# Output files
def save_data(args, **kwargs):
    c = args['C']
    kernel = args['kernel']
    is_report = args['is_print_report']
    is_pred = args['is_print_pred']
    is_train = args['is_train']
    model_name = args['model_name']
    y_test = kwargs['y_test']
    y_pred = kwargs['y_pred']
    test_pred = kwargs['test_pred']

    if is_report:
        write_report(c, kernel, y_test, y_pred, is_train, model_name)

    if is_pred:
        write_pred(c, kernel, test_pred, is_train, model_name)



# Save reports generated from validation data to reflect model conditions
def write_report(c, kernel, y_test, y_pred, is_train, model_name):

    if is_train:
        file = open(RESULT_REPORT_PATH + str(c) + '_' + kernel + '_Report__' + PRESENT_TIME + '.txt', 'w')
    else:
        file = open(RESULT_REPORT_PATH + '[' + model_name + ']_Report__' + PRESENT_TIME + '.txt', 'w')

    file.write(str(confusion_matrix(y_test, y_pred)))
    file.write('\n')
    file.write(str(classification_report(y_test, y_pred)))
    file.write('\n')
    file.write(str(y_pred))
    file.close()

# Save test data classification results
def write_pred(c, kernel, test_pred, is_train, model_name):
    if is_train:
        # file = open(RESULT_DATA_PATH + str(c) + '_' + kernel + '_Out__' + PRESENT_TIME + '.txt', 'w')
        file = open(RESULT_DATA_PATH + 'test-out.txt', 'w')
    else:
        # file = open(RESULT_DATA_PATH + '[' + model_name + ']_Out__' + PRESENT_TIME + '.txt', 'w')
        file = open(RESULT_DATA_PATH + 'test-out.txt', 'w')

    file.write(str(test_pred))
    file.close()


def show_graph(args, **kwargs):
    c = args['C']
    is_roc = args['is_graph_roc']
    is_train = args['is_train']
    kernel = args['kernel']
    model_name = args['model_name']
    y_te = kwargs['y_test']
    y_t_pr = kwargs['y_test_probability']

    if is_roc:
        draw_roc(c, kernel, y_te, y_t_pr, is_train, model_name)

# Drawing, ROC
def draw_roc(c, kernel, y_test, y_test_probability, is_train, model_name):

    # Obtain true/false rate
    fpr, tpr, threshold = roc_curve(y_test, y_test_probability)

    # Mapping data
    ## y_test
    file = open(RESULT_GRAPH_PATH + str(c) + '_' + kernel + "_y_test__" + PRESENT_TIME + '.txt', 'w')
    file.write(str(y_test))
    file.close()

    ## y_probability
    file = open(RESULT_GRAPH_PATH + str(c) + '_' + kernel + "_y_probability" + PRESENT_TIME + '.txt', 'w')
    file.write(str(y_test_probability))
    file.close()

    ## threshold
    file = open(RESULT_GRAPH_PATH + str(c) + '_' + kernel + "_threshold_" + PRESENT_TIME + '.txt', 'w')
    file.write(str(threshold))
    file.close()

    ## fpr
    file = open(RESULT_GRAPH_PATH + str(c) + '_' + kernel + "_fpr_" + PRESENT_TIME + '.txt', 'w')
    file.write(str(fpr))
    file.close()

    ## tpr
    file = open(RESULT_GRAPH_PATH + str(c) + '_' + kernel + "_tpr_" + PRESENT_TIME + '.txt', 'w')
    file.write(str(tpr))
    file.close()


    # Drawing
    plt.ion()  # The key function for the success of enabling interactive mode

    if is_train:
        fig_name = RESULT_GRAPH_PATH + str(c) + '_' + kernel + "_ROC__" + PRESENT_TIME + '.png'
    else:
        fig_name = RESULT_GRAPH_PATH + '[' + model_name + ']_ROC__' + PRESENT_TIME + '.png'

    plt.figure(fig_name, figsize=(6, 6))

    # average time of each eat
    plt.subplot(1, 1, 1)
    plt.title("ROC")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.plot(fpr,tpr , '-g', lw=1)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    # plt.legend(loc="lower right")
    plt.savefig(fig_name)



    # plt.pause(1.001)
    # clear memory
    # plt.clf()  # clear

# for i in range(0,8):
run()
