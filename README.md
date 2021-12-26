# MNIST classification using SVM models

This is the solution of Exercise 7 and 8 of Chapter 7 of the book
*Hands-On Machine Learning with Scikit-Learn, Keras & Tensorflow*.

It classifies digits between 0 and 9. It is trained using the 
[MNIST dataset from OpenML](https://www.openml.org/d/554).

It also provides a simple web interface for testing the predictions,
which can be run with the following command (make sure to install the
dependencies and generate the model before):

    $ flask run --host=0.0.0.0

and looks like this.

![Screenshot of the web interface.](web_screenshot.png)

## Install dependencies

Create a virtual environment, activate it and install the
dependencies with

    $ python3 -m venv venv
    $ source ./venv/bin/activate
    $ pip install -r requirements.txt

## TLDR

To generate and train the model use the following commands:

    $ python -m fetch
    $ python -m train -M svm

## Using

You can use the following models from the CLI:

    $ python -m fetch -h
    usage: fetch.py [-h] [-o OUTPATH] [-s] [-r]
    
    Fetch the mnist_784 dataset from openml, and save it to HDF5.
    
    optional arguments:
      -h, --help            show this help message and exit
      -o OUTPATH, --outpath OUTPATH
                            The filepath where the HDF5 dataset will be saved.
      -s, --shift           Augment the dataset by shifting the images.
      -r, --rotate          Augment the dataset by rotating the images.

    $ python3 -m train --help
    usage: train.py [-h] [-g] [-d DATASET] [-o OUTPATH] [-v] [-m] [-V] [-C REGULARIZATION] [-G GAMMA]
                    [-E ESTIMATORS] [-D MAX_DEPTH] [-L LOSS] [-P PENALTY] [-A ALPHA] [-M MODEL]

    Train a model from an HDF5 dataset and export it to a file.

    optional arguments:
      -h, --help            show this help message and exit
      -g, --gridsearch      Perform a grid search for optimizing the hyperparameters. If specified the model
                            hyperparameters (e.g. --gamma and --regularization) are ignored.
      -d DATASET, --dataset DATASET
                            A filepath containing the dataset in the HDF5 format.
      -o OUTPATH, --outpath OUTPATH
                            The filepath for storing the trained model.
      -v, ---verbose        Display information about the trained model on stdout.
      -m, --metrics         Display metrics computed on the test set.
      -V, --validation      Use the validation set to compute the metrics.
      -C REGULARIZATION, --regularization REGULARIZATION
                            Regularization parameter for SVM classifiers. The strength of the regularization is
                            inversely proportional to C. Must be strictly positive. The penalty is a squared l2
                            penalty
      -G GAMMA, --gamma GAMMA
                            Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. Can be a float value or one of
                            'scale' and 'auto'
      -E ESTIMATORS, --estimators ESTIMATORS
                            Number of estimators used for ensemble classifiers. Can be used with Random Forest or
                            Extra Trees classifiers.
      -D MAX_DEPTH, --max_depth MAX_DEPTH
                            Max depth of tree classifiers. Can be used with Random Forest or Extra Trees
                            classifiers.
      -L LOSS, --loss LOSS  The loss function to be used for SGD classifiers. Can be 'hinge', 'log',
                            'modified_huber', 'squared_hinge', 'perceptron', or a regression loss: 'squared_error',
                            'huber', 'epsilon_insensitive', or 'squared_epsilon_insensitive'.
      -P PENALTY, --penalty PENALTY
                            The penalty (aka regularization term) to be used for SGD classifiers. Can be either 'l1'
                            or 'l2'.
      -A ALPHA, --alpha ALPHA
                            Constant that multiplies the regularization term for SGD classifiers. The higher the
                            value, the stronger the regularization.
      -M MODEL, --model MODEL
                            The algorithm to be used for training the model. The following values are allowed: svm,
                            forest, extra, sgd.


    $ python -m predict -h
    usage: predict.py [-h] [-m MODEL] [-v] [images [images ...]]

    Predict an image of a digit between 0 and 9 using an SVM classificator.

    positional arguments:
      images                The filepath of the image to be predicted.

    optional arguments:
      -h, --help            show this help message and exit
      -m MODEL, --model MODEL
                            The filepath of the trained model.
      -v, ---verbose        Display information about the trained model on stdout.

## Performance

    $ python3 -m train -v -m -V -g -M svm
    INFO    Loading dataset from data.h5
    INFO    Performing grid search.
    Fitting 5 folds for each of 8 candidates, totalling 40 fits
    INFO    Search time: 638.58s
    INFO    Search Accuracy: 0.9448000000000001
    INFO    Best parameters: {
      "clf__C": 8,
      "clf__gamma": "auto",
      "clf__kernel": "rbf"
    }

    INFO    Training an svm classifier.
    INFO    Training time: 226.63s
    INFO    Metrics computed on the test set
                  precision    recall  f1-score   support

              0       0.98      0.99      0.99       985
              1       0.99      0.99      0.99      1110
              2       0.97      0.98      0.97      1004
              3       0.98      0.95      0.97      1000
              4       0.96      0.98      0.97       976
              5       0.97      0.96      0.97       897
              6       0.98      0.97      0.98       969
              7       0.94      0.97      0.96      1062
              8       0.97      0.96      0.97       995
              9       0.97      0.96      0.97      1002

        accuracy                           0.97     10000
      macro avg       0.97      0.97      0.97     10000
    weighted avg       0.97      0.97      0.97     10000

    Confusion matrix:
    [[ 974    1    0    1    0    2    4    3    0    0]
    [   1 1096    3    1    1    0    1    5    1    1]
    [   3    1  980    3    2    1    2    9    3    0]
    [   0    0   11  955    0    8    0   14    9    3]
    [   0    4    2    0  958    0    1    2    0    9]
    [   4    0    1   10    1  861    6    5    7    2]
    [   4    1    2    1    3    8  941    6    3    0]
    [   1    6    5    0   10    0    0 1031    1    8]
    [   3    1    3    5    7    3    5    6  960    2]
    [   2    0    6    1   14    1    0   15    2  961]]


    $ python3 -m train -v -m -V -g -M forest
    INFO    Loading dataset from data.h5
    INFO    Performing grid search.
    Fitting 5 folds for each of 6 candidates, totalling 30 fits
    INFO    Search time: 42.32s
    INFO    Search Accuracy: 0.9445
    INFO    Best parameters: {
      "clf__max_depth": 64,
      "clf__n_estimators": 100,
      "clf__n_jobs": -1
    }

    INFO    Training an forest classifier.
    INFO    Training time: 13.31s
    INFO    Metrics computed on the test set
                  precision    recall  f1-score   support

               0       0.98      0.99      0.99       985
               1       0.98      0.99      0.99      1110
               2       0.96      0.97      0.96      1004
               3       0.96      0.94      0.95      1000
               4       0.96      0.97      0.97       976
               5       0.97      0.96      0.96       897
               6       0.98      0.98      0.98       969
               7       0.97      0.97      0.97      1062
               8       0.95      0.95      0.95       995
               9       0.95      0.95      0.95      1002

        accuracy                           0.97     10000
       macro avg       0.97      0.97      0.97     10000
    weighted avg       0.97      0.97      0.97     10000

    Confusion matrix:
    [[ 974    1    0    0    3    1    2    0    3    1]
     [   1 1098    2    0    2    0    2    4    1    0]
     [   5    1  973    3    3    0    0    5   14    0]
     [   1    1   17  943    0   10    0   12   10    6]
     [   0    4    0    0  947    0    4    1    2   18]
     [   5    2    4    8    0  858    8    0    6    6]
     [   1    0    0    0    2    9  952    0    5    0]
     [   0    5    8    1    9    0    0 1025    3   11]
     [   1    5    6    8    5    2    6    2  950   10]
     [   4    1    3   16   13    2    0    8    4  951]]

    $ python3 -m train -v -m -g -M extra
    INFO    Loading dataset from data.h5
    INFO    Performing grid search.
    Fitting 5 folds for each of 6 candidates, totalling 30 fits
    INFO    Search time: 46.13s
    INFO    Search Accuracy: 0.9484999999999999
    INFO    Best parameters: {
      "clf__max_depth": 16,
      "clf__n_estimators": 100,
      "clf__n_jobs": -1
    }

    INFO    Training an extra classifier.
    INFO    Training time: 13.89s
    INFO    Metrics computed on the test set
                  precision    recall  f1-score   support

               0       0.97      0.99      0.98       980
               1       0.98      0.99      0.98      1135
               2       0.96      0.96      0.96      1032
               3       0.96      0.96      0.96      1010
               4       0.97      0.96      0.97       982
               5       0.97      0.96      0.96       892
               6       0.97      0.98      0.97       958
               7       0.97      0.95      0.96      1028
               8       0.97      0.95      0.96       974
               9       0.94      0.95      0.94      1009

        accuracy                           0.97     10000
       macro avg       0.97      0.97      0.97     10000
    weighted avg       0.97      0.97      0.97     10000

    Confusion matrix:
    [[ 969    1    0    0    0    2    4    1    3    0]
     [   0 1123    1    5    0    1    3    0    1    1]
     [   7    1  995    4    4    0    5   10    6    0]
     [   0    0   10  968    0   13    0   11    5    3]
     [   1    0    2    0  945    0    6    0    2   26]
     [   3    3    0   13    1  857    7    1    5    2]
     [   5    3    0    1    3    4  939    0    3    0]
     [   2    8   20    2    1    0    0  976    2   17]
     [   4    0    5    5    7    7    4    5  927   10]
     [   6    7    1   11   15    3    1    5    6  954]]


    $ python3 -m train -v -m -g -M sgd
    INFO    Loading dataset from data.h5
    INFO    Performing grid search.
    Fitting 5 folds for each of 12 candidates, totalling 60 fits
    /home/giacomo/.local/lib/python3.8/site-packages/sklearn/linear_model/_stochastic_gradient.py:574: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
      warnings.warn("Maximum number of iteration reached before "
    /home/giacomo/.local/lib/python3.8/site-packages/sklearn/linear_model/_stochastic_gradient.py:574: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
      warnings.warn("Maximum number of iteration reached before "
    /home/giacomo/.local/lib/python3.8/site-packages/sklearn/linear_model/_stochastic_gradient.py:574: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
      warnings.warn("Maximum number of iteration reached before "
    /home/giacomo/.local/lib/python3.8/site-packages/sklearn/linear_model/_stochastic_gradient.py:574: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
      warnings.warn("Maximum number of iteration reached before "
    /home/giacomo/.local/lib/python3.8/site-packages/sklearn/linear_model/_stochastic_gradient.py:574: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
      warnings.warn("Maximum number of iteration reached before "
    /home/giacomo/.local/lib/python3.8/site-packages/sklearn/linear_model/_stochastic_gradient.py:574: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
      warnings.warn("Maximum number of iteration reached before "
    /home/giacomo/.local/lib/python3.8/site-packages/sklearn/linear_model/_stochastic_gradient.py:574: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
      warnings.warn("Maximum number of iteration reached before "
    /home/giacomo/.local/lib/python3.8/site-packages/sklearn/linear_model/_stochastic_gradient.py:574: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
      warnings.warn("Maximum number of iteration reached before "
    /home/giacomo/.local/lib/python3.8/site-packages/sklearn/linear_model/_stochastic_gradient.py:574: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
      warnings.warn("Maximum number of iteration reached before "
    /home/giacomo/.local/lib/python3.8/site-packages/sklearn/linear_model/_stochastic_gradient.py:574: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
      warnings.warn("Maximum number of iteration reached before "
    /home/giacomo/.local/lib/python3.8/site-packages/sklearn/linear_model/_stochastic_gradient.py:574: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
      warnings.warn("Maximum number of iteration reached before "
    /home/giacomo/.local/lib/python3.8/site-packages/sklearn/linear_model/_stochastic_gradient.py:574: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
      warnings.warn("Maximum number of iteration reached before "
    INFO    Search time: 938.60s
    INFO    Search Accuracy: 0.8974
    INFO    Best parameters: {
      "clf__alpha": 0.0001,
      "clf__loss": "hinge",
      "clf__n_jobs": -1,
      "clf__penalty": "l2"
    }

    INFO    Training an sgd classifier.
    INFO    Training time: 122.90s
    INFO    Metrics computed on the test set
                  precision    recall  f1-score   support

              0       0.96      0.96      0.96       980
              1       0.98      0.95      0.97      1135
              2       0.94      0.87      0.90      1032
              3       0.92      0.88      0.90      1010
              4       0.94      0.90      0.92       982
              5       0.90      0.82      0.86       892
              6       0.93      0.93      0.93       958
              7       0.95      0.91      0.93      1028
              8       0.66      0.93      0.77       974
              9       0.92      0.86      0.89      1009

        accuracy                           0.90     10000
      macro avg       0.91      0.90      0.90     10000
    weighted avg       0.91      0.90      0.90     10000

    Confusion matrix:
    [[ 940    0    0    0    0    9    6    1   24    0]
    [   0 1083    6    1    0    3    4    0   38    0]
    [   5    4  898   13    7    1   15    6   80    3]
    [   4    1   13  884    1   23    3    5   66   10]
    [   1    0    6    0  882    2    8    3   55   25]
    [   4    2    0   38    9  734   17    7   71   10]
    [   8    2    9    0    8   16  888    1   26    0]
    [   3    2   18    4    6    1    0  932   35   27]
    [   5    4    4   15    3   23    9    2  904    5]
    [   5    5    0    8   27    6    0   23   64  871]]

    $ python3 -m train -v -m -M voting
    INFO    Loading dataset from data.h5
    INFO    Training an voting classifier.
    INFO    Training time: 1263.06s
    INFO    Metrics computed on the test set
                  precision    recall  f1-score   support

               0       0.97      0.99      0.98       980
               1       0.98      0.99      0.99      1135
               2       0.96      0.97      0.97      1032
               3       0.97      0.97      0.97      1010
               4       0.97      0.97      0.97       982
               5       0.98      0.96      0.97       892
               6       0.98      0.98      0.98       958
               7       0.97      0.96      0.97      1028
               8       0.97      0.96      0.97       974
               9       0.96      0.95      0.95      1009

        accuracy                           0.97     10000
       macro avg       0.97      0.97      0.97     10000
    weighted avg       0.97      0.97      0.97     10000

    Confusion matrix:
    [[ 970    0    1    0    0    2    3    1    3    0]
     [   0 1126    2    3    0    1    2    0    1    0]
     [   6    1 1001    4    2    0    3   10    5    0]
     [   0    0    7  981    0    7    0    8    5    2]
     [   1    0    2    0  952    0    4    0    3   20]
     [   3    1    2    9    4  860    6    1    4    2]
     [   6    3    1    0    3    4  938    0    3    0]
     [   1    7   18    1    1    0    0  987    1   12]
     [   4    0    5    6    4    4    3    5  936    7]
     [   7    7    2   12   14    3    0    5    3  956]]
