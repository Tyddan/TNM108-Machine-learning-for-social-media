import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import statistics
from math import pi
from math import e

iris = datasets.load_iris()
data = iris.data
target = iris.target
target_values = np.unique(target)
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.3)

# hej jag är en kommentar.


def group_by_class(self, data, target):
    """ 
    :param data: Training set
    :param target: the list of class labels labelling data 
    :return: Separate the data by their target class; that is, 
    create one group for every value of the target class. 
    It returns all the groups """
    separated = [[x for x, t in zip(data, target) if t == c]
                 for c in self.target_values]
    groups = [np.array(separated[0]), np.array(
        separated[1]), np.array(separated[2])]
    return np.array(groups)


""" The probability of each group of instances (that is the class) 
with respect to the total number of instances """
#len(group)/len(data)


# learn from train set by calculating the mean and standard deviation
def train(self, data, target):
    """ 
    :param data: a dataset 
    :param target: the list of class labels labelling data 
    :return: For each target class: 1. yield prior_prob: the probability 
    of each class 2. yield summary: list of {'mean': 0.0,'stdev': 0.0} 
    for every feature in data """
    groups = self.group_by_class(data, target)
    for index in range(groups.shape[0]):
        group = groups[index]
        self.summaries[self.target_values[index]] = {
            'prior_prob': len(group)/len(data),
            'summary': [i for i in self.summarize(group)]
        }

# Product of all normal probabilities


def normal_pdf(self, x, mean, stdev):
    """
    :param x: the value of a feature F 
    :param mean: μ - average of F 
    :param stdev: σ - standard deviation of F 
    :return: Gaussian (Normal) Density function. 
    N(x; μ, σ) = (1 / 2πσ) * (e ^ (x–μ)^2/-2σ^2 """

    variance = stdev ** 2
    exp_squared_diff = (x - mean) ** 2
    exp_power = -exp_squared_diff / (2 * variance)
    exponent = e ** exp_power
    denominator = ((2 * pi) ** .5) * stdev
    normal_prob = exponent / denominator
    return normal_prob


# Product of the prior Probability and the Likelihood
def joint_probabilities(self, data): 
    """
    :param data: dataset in a matrix form (rows x col) 
    :return: Use the normal_pdf(self, x, mean, stdev) to 
    calculate the Normal Probability for each feature 
    Yields the product of all Normal Probabilities and the 
    Prior Probability of the class. """ 
    joint_probs = {} 
    for y in range(self.target_values.shape[0]):
        target_v = self.target_values[y] 
        item = self.summaries[target_v] 
        total_features = len(item['summary']) 
        likelihood = 1 
        for index in range(total_features): 
            feature = data[index] 
            mean = self.summaries[target_v]['summary'][index]['mean'] 
            stdev = self.summaries[target_v]['summary'][index]['stdev']**2 
            normal_prob = self.normal_pdf(feature, mean, stdev) 
            likelihood *= normal_prob 
        prior_prob = self.summaries[target_v]['prior_prob'] 
        joint_probs[target_v] = prior_prob * likelihood 
    return joint_probs


def marginal_pdf(self, joint_probabilities): 
    """ 
    :param joint_probabilities: list of joint probabilities for each feature 
    :return: Marginal Probability Density Function (Predictor Prior Probability) 
    Joint Probability = prior * likelihood Marginal Probability is 
    the sum of all joint probabilities for all classes """ 
    marginal_prob = sum(joint_probabilities.values()) return marginal_prob
