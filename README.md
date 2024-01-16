# bayesian_classification
This script performs classification on a synthetic dataset using Bayesian classification, estimating error probabilities, and comparing results with the average risk minimization rule. This project was made for my Introduction to Machine Learning class.

Data Generation:
  Two classes are generated with mean vectors m1 and m2 and a common covariance matrix S.
  The number of data points per class is defined by n_points_per_class.
  Data points are generated using the np.random.multivariate_normal function.
  The script labels the data points and plots the dataset.

Bayesian Classification:
  A priori probabilities P1 and P2 are estimated based on the number of data points per class.
  Probability density functions (pdf) are estimated for each data point using the covariance matrix S.
  Data points are classified based on Bayes' theorem, and error probability (Pe) is calculated.

Average Risk Minimization Rule:
  A loss matrix L is defined.
  Probability density functions (pdf) are re-estimated.
  Data points are classified based on the average risk minimization rule.
  Average risk (Ar) is calculated.

Results and Plots:
  The script prints the probability of error (Pe) and the average risk (Ar).
  Three separate plots are created to visualize the classification results:
  Bayesian classification results.
  Average risk minimization rule results.
  Original dataset with both classifications.

The script concludes that the average risk minimization rule yields a smaller average risk compared to the probability of error obtained by the classic Bayesian classification rule. The reason is explained: in the average risk minimization rule, classification errors on data stemming from class w2 are considered "cheaper" compared to the opposite classification errors.

Note: The script uses np.concatenate, np.zeros, and reduce from the NumPy library for array manipulation and calculations. The plt.plot function from Matplotlib is used for data visualization.
