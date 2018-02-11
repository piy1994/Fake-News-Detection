"""
This is the script I used to perform the grid search. 

"""

import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, LSTM, Embedding, Reshape
from keras.layers import Bidirectional
from keras.utils import np_utils

from keras.layers import Conv1D,MaxPooling1D
from sklearn.metrics import accuracy_score

from sklearn.utils.class_weight import compute_class_weight
import itertools

from keras.callbacks import ModelCheckpoint  
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


"""
This functions reads the processed data empeddings and divides the data into train and test validation set.
"""
def loadInputData():
    data=pickle.load(open("processed_data_embed.p","rb"))
    x = []
    y = []

    for data_row in data:
        x.append(data_row[0].tolist())
        y.append(data_row[1])

    x = x[1:]
    y = y[1:]
    x = np.array(x)
    x_dummy = np.reshape(x,(len(x),50,1))
    dummy_y = np_utils.to_categorical(y)
    X_train, X_test, y_train, y_test = train_test_split(x_dummy, dummy_y, test_size=0.2, random_state=42)
    return X_train,X_test,y_train,y_test,y,dummy_y

"""
This function creates the keras model for the input parameters.
"""

def create_model(num_classes,num_lstm_units,X_train_shape,kernel_size):
	# create model
	model = Sequential()
	model.add(Bidirectional(LSTM(units = num_lstm_units ,return_sequences=True),
                         input_shape=(X_train_shape[1],X_train_shape[2])))

	model.add(Conv1D(filters=16, kernel_size=kernel_size, padding='same', activation='relu'))
	model.add(MaxPooling1D(pool_size=2))
    
	model.add(Conv1D(filters=32, kernel_size=kernel_size, padding='same', activation='relu'))
	model.add(MaxPooling1D(pool_size=2))
    
	model.add(Conv1D(filters=64, kernel_size=kernel_size, padding='same', activation='relu'))
	model.add(MaxPooling1D(pool_size=2))
    
	model.add(Dropout(0.3))
	model.add(Flatten())
    
	model.add(Dense(150, activation='relu'))
	model.add(Dropout(0.4))
    
	model.add(Dense(num_classes,activation = 'softmax'))
    
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
	model.summary()
	return model

"""
I need to decide the parametes: I can take either 3x3 or 5x5 or 7x7 as the kernel dimesion, for
lstm i can take 32 or either 64 
keep patience at 10 , its more than enought
Parametes i am trying to optimize in the following function are num_lstm_units and kernel_size
"""

def gridSearchLstmKer(X_train,X_test,y_train,y_test,y,dummy_y):
    lstm_units = [32,64]
    kernels = [3,5,7]

    """constant parameters"""
    num_classes = dummy_y.shape[1]
    class_weight = compute_class_weight('balanced', np.unique(y), y)
    num_classes = dummy_y.shape[1]
    input_shape = X_train.shape

    grid_search_lst = list(itertools.product(*[lstm_units,kernels]))

    batch_size = 100
    epochs = 100

    for lst,ker in grid_search_lst:
        
        model = create_model(num_classes =num_classes,num_lstm_units = lst,
                             X_train_shape = input_shape,kernel_size = ker)
        
        E_Stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto') 
        
        checkpointer = ModelCheckpoint(filepath='results/{0}_{1}.weights.best.hdf5'.format(lst,ker),
                                       verbose=1,save_best_only=True)
        
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, 
                  validation_data=(X_test, y_test),class_weight = class_weight,
                  callbacks=[checkpointer,E_Stop],shuffle = True)


    # now lets test things and output the results into the text file.

    class_weight_norm = class_weight/np.linalg.norm(class_weight)

    f = open('out_inverse_and_mean.txt', 'w')
    for lst,ker in grid_search_lst:
        f.write('For the model with lstm units = {0} and kernels = {1} \n'.format(lst,ker))

        model = create_model(num_classes =num_classes,num_lstm_units = lst,X_train_shape = input_shape,kernel_size = ker)
        model.load_weights('results/{0}_{1}.weights.best.hdf5'.format(lst,ker))
        score = model.evaluate(X_test,y_test,verbose = 0)
        f.write('The total loss is {0} and the average accuracy is {1} \n'.format(score[0],score[1]))

        y_pred = np_utils.to_categorical(model.predict_classes(X_test))
        precision, recall,fscore,_ = precision_recall_fscore_support(y_test,y_pred)
        f.write('The precision is {0} and average precision is {1}\n'.format(precision,precision.mean()))
        f.write('The recall is {0} and the average recall is {1}\n'.format(recall,recall.mean()))
        f.write('The fscore is {0} , the average fscore is {1}, the inverse weighted fscore is {2}'.format(
                fscore,fscore.mean(),np.inner(class_weight_norm,fscore)))
        
        f.write('\n \n \n')
        
        
    f.close()


"""
This function perform the gridSearch over the batch size
"""
def gridSearchBatchSize(X_train,X_test,y_train,y_test,y,dummy_y,lstm,ker):

    num_classes = dummy_y.shape[1]
    input_shape = X_train.shape
    class_weight = compute_class_weight('balanced', np.unique(y), y)
    class_weight_norm = class_weight/np.linalg.norm(class_weight)

    batch_size_lst = [64,128,256]


    for  batch in batch_size_lst:

        if Path('results/batch/{0}_{1}_{2}.weights.best.hdf5'.format(lstm,ker,batch)).is_file():
            continue

        model = create_model(num_classes =num_classes,num_lstm_units = lstm,
                             X_train_shape = input_shape,kernel_size = ker)
        
        E_Stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto') 
        
        checkpointer = ModelCheckpoint(filepath='results/batch/{0}_{1}_{2}.weights.best.hdf5'.format(lstm,ker,batch),
                                       verbose=1,save_best_only=True)
        
        model.fit(X_train, y_train, batch_size=batch, epochs=epochs, 
                  validation_data=(X_test, y_test),class_weight = class_weight,
                  callbacks=[checkpointer,E_Stop],shuffle = True)

    # Training is done, now let's test all of them in put the results in a file 

    f = open('out_inverse_and_mean_batch.txt', 'w')

    for b in batch_size_lst:
        f.write('For the model with lstm units = {0} and kernels = {1} and batch size {2}\n'.format(lstm,ker,b))

        model = create_model(num_classes =num_classes,num_lstm_units = lstm,X_train_shape = input_shape,kernel_size = ker)
        model.load_weights('results/batch/{0}_{1}_{2}.weights.best.hdf5'.format(lstm,ker,b))
        score = model.evaluate(X_test,y_test,verbose = 0)
        f.write('The total loss is {0} and the average accuracy is {1} \n'.format(score[0],score[1]))

        y_pred = np_utils.to_categorical(model.predict_classes(X_test))
        precision, recall,fscore,_ = precision_recall_fscore_support(y_test,y_pred)
        
        f.write('The precision is {0} and average precision is {1}\n'.format(precision,precision.mean()))
        f.write('The recall is {0} and the average recall is {1}\n'.format(recall,recall.mean()))
        f.write('The fscore is {0} , the average fscore is {1}, the inverse weighted fscore is {2}'.format(
                fscore,fscore.mean(),np.inner(class_weight_norm,fscore)))
        
        f.write('\n \n \n')
        
        
    f.close()



# now evaluating the model with the optimzied parameters and plot the training and testing cureves
def plotTrainTestCurves(X_train,X_test,y_train,y_test,y,dummy_y,lstm,ker,batch)


    num_classes = dummy_y.shape[1]
    input_shape = X_train.shape
    class_weight = compute_class_weight('balanced', np.unique(y), y)

    epochs = 100
    #batch_size = 128
    batch_size = batch
    model = create_model(num_classes =num_classes,num_lstm_units = lstm,
                         X_train_shape = input_shape,kernel_size = ker)

    E_Stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto') 

    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, 
              validation_data=(X_test, y_test),class_weight = class_weight,
              callbacks=[E_Stop],shuffle = True)

    #print(history.history)


    #%% 
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('results/model_accuracy.png')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('results/model_loss.png')
    plt.show()


"""
This function performs the jittering on the test data to see how the model gets affected by noise 
"""

def checkingModelStabality(X_train,X_test,y_train,y_test,y,dummy_y,lstm,ker,batch)

    num_classes = dummy_y.shape[1]
    input_shape = X_train.shape
    class_weight = compute_class_weight('balanced', np.unique(y), y)
    class_weight_norm = class_weight/np.linalg.norm(class_weight)

    f = open('out_peerturb.txt', 'w')
    for frac in [0.05,0.0001,0.001,0.01]:
        f.write('Noise fraction added = {0}'.format(frac))
        
        X_test_noised = X_test.reshape(X_test.shape[0],50).copy()
        noise_add = np.random.rand(X_test_noised.shape[0],50)*frac
        
        X_test_noised += noise_add
        X_test_noised = X_test_noised.reshape(X_test.shape[0],50,1)
        
        
        model = create_model(num_classes =num_classes,num_lstm_units = lstm,X_train_shape = input_shape,kernel_size = ker)
        model.load_weights('results/batch/{0}_{1}_{2}.weights.best.hdf5'.format(lstm,ker,batch))
        score = model.evaluate(X_test_noised,y_test,verbose = 0)
        f.write('The total loss is {0} and the average accuracy is {1} \n'.format(score[0],score[1]))
        
        y_pred = model.predict_classes(X_test_noised)
        y_test_lab = [np.where(r==1)[0][0] for r in y_test]
        precision, recall,fscore,_ = precision_recall_fscore_support(y_test_lab,y_pred)
        
        f.write('The precision is {0} and average precision is {1}\n'.format(precision,precision.mean()))
        f.write('The recall is {0} and the average recall is {1}\n'.format(recall,recall.mean()))
        f.write('The fscore is {0} , the average fscore is {1}, the inverse weighted fscore is {2}'.format(
                fscore,fscore.mean(),np.inner(class_weight_norm,fscore)))
        f.write('\n \n \n')
    f.close()



"""
This function prints and plots the confusion matrix.
Normalization can be applied by setting `normalize=True`.
"""
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('Confusion_matrix_{0}'.format('normalized' if normalize is True else 'non_normalized'))


# Making the confusion matrix

def makeConfusionMatrix(X_train,X_test,y_train,y_test,y,dummy_y,lstm,ker,batch)

    num_classes = dummy_y.shape[1]
    input_shape = X_train.shape
    class_weight = compute_class_weight('balanced', np.unique(y), y)
    class_weight_norm = class_weight/np.linalg.norm(class_weight)


    model = create_model(num_classes =num_classes,num_lstm_units = lstm,X_train_shape = input_shape,kernel_size = ker)
    model.load_weights('results/batch/{0}_{1}_{2}.weights.best.hdf5'.format(lstm,ker,batch))
    y_pred = model.predict_classes(X_test)
    y_test_lab = [np.where(r==1)[0][0] for r in y_test]

    
    cnf_mat = confusion_matrix(y_test_lab, y_pred)
    classes = ["unrelated","discuss","agree","disagree"]

    plt.figure()
    plot_confusion_matrix(cnf_mat, classes=classes,title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_mat, classes=classes, normalize=True,title='Normalized confusion matrix')


if __name__ == '__main__':
    #first thing is to load the data and split it into test and train

    X_train,X_test,y_train,y_test,y,dummy_y = loadInputData()

    # Perform a Grid search on the number of Bi-lstm units and Number of kernels
    # The function produces the output file out_inverse_and_mean.txt . View it and decide the parameters

    gridSearchLstmKer(X_train,X_test,y_train,y_test,y,dummy_y)

    # Now optimize the batch size. Results will be in the txt file : out_inverse_and_mean_batch

    lstm =  int(input("view the file and decide on the number of Bi-lstm units"))
    ker = int(input("view the file and decide on the dimension of the Kernels"))

    gridSearchBatchSize(X_train,X_test,y_train,y_test,y,dummy_y,lstm,ker)

    batch = int(input("view the file and decide on the batch size"))

    # We have now optimzed the Bilstm, kernels and the batch size

    # Plot the training curves and testing curves

    plotTrainTestCurves(X_train,X_test,y_train,y_test,y,dummy_y,lstm,ker,batch)


    # check how stable the model is with respect to the noise

    checkingModelStabality(X_train,X_test,y_train,y_test,y,dummy_y,lstm,ker,batch)

    # As a final step, make the confusion matrix,

    makeConfusionMatrix(X_train,X_test,y_train,y_test,y,dummy_y,lstm,ker,batch)