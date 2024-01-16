
import numpy as np
import math
from functools import reduce
from matplotlib import pyplot as plt


def classification_7_19():
    np.random.seed(0)
    # Definition of mu's and Sigma
    # Mean vectors and covariance matrix
    
    #Here we are assigning the mean vectors which are modeled by gaussian distribution
    m1 = np.array([[0], [0]]) 
    m2 = np.array([[2], [2]])
    #Here we are defining the common covariance matrix which is a summation of the matric with the following points below
    S = np.array([[1, .25], [.25, 1]])
    # Number of data points

    #Here we are defining the number of points to plot the data set X. Since the classes will have the same number of points, we can use this as a constant.
    n_points_per_class = 500

    # (i) Data point generation
    
    #Here we are assigning a random number to one of the points per class to each respective point on the mean vectors
    X = np.concatenate((np.random.multivariate_normal(m1.flatten().conj().T, S, n_points_per_class),
                        np.random.multivariate_normal(m2.flatten().conj().T, S, n_points_per_class)), axis=0).conj().T

    #Here we are labeling the variables so that it can define on which mean vector is which point
    label = np.concatenate((np.ones((1, n_points_per_class)),
                            2 * np.ones((1, n_points_per_class))), axis=1)
    [l, p] = X.shape
    # Plot the data set
    
    #Here we are plotting the data set of the mean vector 1 and 2
    plt.figure(1)
    plt.plot(X[0, np.nonzero(label == 1)], X[1, np.nonzero(label == 1)], '.b')
    plt.plot(X[0, np.nonzero(label == 2)], X[1, np.nonzero(label == 2)], '.r')

    # (ii) Bayes classification of X
    # Estimation of a priori probabilities

    #Here we are assigning values: P1 refers to the number of points per class (500) divided by the numbers in the dataset, P2 refers to just P1 itself, p1 refers to an array of size p filled with zeroes, p2 refers the same as p1 
    P1 = n_points_per_class/p
    P2 = P1
    p1 = np.zeros(p)
    p2 = np.zeros(p)
    # Estimation of pdf's for each data point

    #Here we are performing a for loop for every element from 1 to p to estimate and add the pdf for each data point according to the baye's theorem
    for i in range(0, p):  # =1:p
        p1[i] = (1/(2*np.pi*np.sqrt(np.linalg.det(S)))) *\
                math.exp(reduce(np.dot, [-(np.array(X[:, i], ndmin=2).conj().T - m1).conj().T, np.linalg.inv(S),
                                         (np.array(X[:, i], ndmin=2).conj().T - m1)]))
        p2[i] = (1/(2*np.pi*np.sqrt(np.linalg.det(S)))) *\
                math.exp(reduce(np.dot, [-(np.array(X[:, i], ndmin=2).conj().T - m2).conj().T, np.linalg.inv(S),
                                         (np.array(X[:, i], ndmin=2).conj().T-m2)]))

    # Classification of the data points

    #Here we are classifiying the datapoints accordind to the parameters, if the product of the number of points per class per the total numbers in the dataset by the pdf of each data point is grater than the product of the number of points per class per the total numbers in the dataset by the pdf of each data point of the other class
    #Then the array callsed classes will contain a one, else it will contain a 2
    classes = np.zeros(p)
    for i in range(0, p):  # =1:p
        if P1*p1[i] > P2*p2[i]:
            classes[i] = 1
        else:
            classes[i] = 2

    # (iii) Error probability estimation

    #Here we are revising for an error probablity estimation in which if the element of the array classes is not equal to the label of coordinate (0,i), then we will get a coint for the probability error estimation. This in order for us to calculate the estimation error.
    Pe = 0  # Probability of error
    for i in range(0, p):  # =1:p
        if classes[i] != label[0][i]:
            Pe += 1
    
    #Here at the end we are dividing the estimation error percentage by the total number of datapoints and print
    Pe /= p
    print('Pe: %f' % Pe)

    # Plot the data set

    #Here we are plotting the data set again with the estimation errors classified
    plt.figure(2)
    plt.plot(X[0, np.nonzero(classes == 1)], X[1, np.nonzero(classes == 1)], '.b')
    plt.plot(X[0, np.nonzero(classes == 2)], X[1, np.nonzero(classes == 2)], '.r')

    # Definition of the loss matrix
    #Here we are defininf the loss matrix to use it on step iv
    L = np.array([[0, 1], [.005, 0]])

    # (iv) Classification of data points according to the average risk
    # minimization rule
    # Estimation of pdf's for each data point

    #Here we are classifying the datapoints in the loss matrix to each respective class by utilizing the risk minimization rule 
    for i in range(0, p):  # =1:p
        p1[i] = (1 / (2 * np.pi * np.sqrt(np.linalg.det(S)))) * \
                math.exp(reduce(np.dot, [-(np.array(X[:, i], ndmin=2).conj().T - m1).conj().T, np.linalg.inv(S),
                                         (np.array(X[:, i], ndmin=2).conj().T - m1)]))
        p2[i] = (1 / (2 * np.pi * np.sqrt(np.linalg.det(S)))) * \
                math.exp(reduce(np.dot, [-(np.array(X[:, i], ndmin=2).conj().T - m2).conj().T, np.linalg.inv(S),
                                         (np.array(X[:, i], ndmin=2).conj().T - m2)]))

    # Classification of the data points

    #Here we are classifying the datapoints again according to the loss matrix and the datapoint that we have for each class
    classes_loss = np.zeros(p)
    for i in range(0, p):  # =1:p
        if L[0][1] * P1 * p1[i] > L[1][0] * P2 * p2[i]:
            classes_loss[i] = 1
        else:
            classes_loss[i] = 2

    # (v) Error probability estimation

    #Here we are calculating the average risk based on the loss matrix and the calculations that we created based on the previous question.  
    Ar = 0  # Average risk
    for i in range(0, p):  # =1:p
        if classes_loss[i] != label[0][i]:
            if label[0][i] == 1:
                Ar = Ar + L[0, 1]
            else:
                Ar = Ar + L[1, 0]
    #Here we are printing the percentage of the average risk based on the number of datasets
    Ar /= p
    print('Ar: %f' % Ar)

    # Plot the data set
    
    #Here we are plotting the datasets of the classes with the average error, the loss matrix and both classes.
    plt.figure(3)
    plt.plot(X[0, np.nonzero(classes_loss == 1)], X[1, np.nonzero(classes_loss == 1)], '.b')
    plt.plot(X[0, np.nonzero(classes_loss == 2)], X[1, np.nonzero(classes_loss == 2)], '.r')

    plt.show()


if __name__ == '__main__':
    classification_7_19()


# The average risk minimization rule leads to smaller value of the average risk compared to the
# probability of error obtained by the classic Bayes classification rule.
# In the former case the classification rule decides for almost all the
# overapping region between the two classes, in favor of \omega_1. This is
# due to the fact that a classification error on data steming from \omega_2 is ``cheap'',
# compared to an opposite classification error.

