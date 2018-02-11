Fake News Detection

myReport.pdf that explains the whole project.

Following are the libraries used

1. numpy
2. Pickle
3. Sklearn
4. Keras
5. itertools
6. matplotlib
7. csv
8. sys

The input dataset is within the two files : train_stances.csv and train_bodies.csv
glove.6B.50d.txt is the wordVector txt file which is used to make the word vectors

There's an Ipython notebook in the dataExploration folder that explores the data

The python file that generates the wordEmbeddings is makingEmbedding.py . It doesn't need any input, just make sure 
that the two training files are present in the same folder where this file is.This file dumps the processed data to a file 
called as processed_data_embed.p .

classification_gridsearch.py is the main file that does all the trainign,grid search and plots the graphs.It generates 3 txt files
out_perturb.txt, out_inverse_and_mean_batch.txt and out_inverse_and_mean.txt . These files contains the result for the grid search.
It also dumps the keras model weights in the result folder. 
This file also makes training and testing loss and accuracy curves which are saved in the result folder.
This files makes plot of confusion matrix which is saved in the current directory.

I ran everything on an aws instance and it took more than 14 hours for the complete grid search.

The baseline model provided by the competetion is included in the fnc-1-baseline folder.
Simply run the python file, fnc_kfold.py and it will output their result in the out_baseline.txt file in the parent folder.




