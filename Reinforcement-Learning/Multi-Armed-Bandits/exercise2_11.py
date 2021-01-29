# -*- coding: utf-8 -*-
from bandit import Bandit
from tqdm import tqdm
import matplotlib.pyplot as plt


def epsilonGreedy(runs: int, timeStep: int, parameters: list):
    """
    parameter study for epsilon-greedy method
    :param runs:
    :param timeStep:
    :param parameters:
    :return: a list of reward correspond to each parameter
    """
    print("Staring epsilon-greedy method...")
    averageRewardInParams = []
    for epsilon in tqdm(parameters):
        averageReward = 0
        for run in range(runs):
            bandit = Bandit(timeStep, epsilon=pow(2, epsilon))
            reward = bandit.epsilonGreedy()  # average reward over time step
            averageReward = averageReward + (reward - averageReward) / (run + 1)
        averageRewardInParams.append(averageReward)
    return averageRewardInParams


def UCB(runs: int, timeStep: int, parameters: list):
    """
    parameter study for Upper-Confidence-Bound Action Selection
    :param runs:
    :param timeStep:
    :param parameters:
    :return: a list of reward correspond to each parameter
    """
    print("Staring UCB method...")
    averageRewardInParams = []
    for c in tqdm(parameters):
        averageReward = 0
        for run in range(runs):
            bandit = Bandit(timeStep, c=pow(2, c))
            reward = bandit.UCB()
            averageReward = averageReward + (reward - averageReward) / (run + 1)
        averageRewardInParams.append(averageReward)
    return averageRewardInParams


def optimisticInitValues(runs: int, timeStep: int, parameters: list):
    """
    parameter study for Optimistic Initial Values
    :param runs:
    :param timeStep:
    :param parameters:
    :return: a list of reward correspond to each parameter
    """
    print("Staring Optimistic Initial Values method...")
    averageRewardInParams = []
    for optValue in tqdm(parameters):
        averageReward = 0
        for run in range(runs):
            bandit = Bandit(timeStep, optValue=pow(2, optValue))
            reward = bandit.optimisticInitValues()
            averageReward = averageReward + (reward - averageReward) / (run + 1)
        averageRewardInParams.append(averageReward)
    return averageRewardInParams


def gradientBandit(runs: int, timeStep: int, parameters: list):
    """
    parameter study for Gradient Bandit Algorithms
    :param runs:
    :param timeStep:
    :param parameters:
    :return: a list of reward correspond to each parameter
    """
    print("Staring Gradient Bandit Algorithms method...")
    averageRewardInParams = []
    for alpha in tqdm(parameters):
        averageReward = 0
        for run in range(runs):
            bandit = Bandit(timeStep, alpha=pow(2, alpha), baseLine=True)
            reward = bandit.gradientBandit()
            averageReward = averageReward + (reward - averageReward) / (run + 1)
        averageRewardInParams.append(averageReward)
    return averageRewardInParams


def lineChart(X, Y, colors, labels):
    """
    :param X: [[]]
    :param Y: [[]]
    :param colors: []
    :param labels: []
    """
    for x, y, c, label in zip(X, Y, colors, labels):
        plt.plot(x, y, color=c, label=label)
    plt.xlabel('2^x')
    plt.ylabel('Average reward over time step')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    runs = 2000
    timeStep = 1000
    parameters = [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2]

    X, Y = [], []
    # e-greedy method
    epsilonGreedyAverageReward = epsilonGreedy(runs, timeStep, parameters[:6])
    Y.append(epsilonGreedyAverageReward)
    X.append(parameters[:6])

    # Upper-Confidence-Bound Action Selection
    UCBaverageReward = UCB(runs, timeStep, parameters[3:])
    Y.append(UCBaverageReward)
    X.append(parameters[3:])

    # Optimistic Initial Values
    OPTaverageReward = optimisticInitValues(runs, timeStep, parameters[5:])
    Y.append(OPTaverageReward)
    X.append(parameters[5:])

    # Gradient Bandit Algorithms
    gradientBanditAverageReward = gradientBandit(runs, timeStep, parameters[2:])
    Y.append(gradientBanditAverageReward)
    X.append(parameters[2:])

    # plot line chart
    labels = ["e-greedy", "UCB", "greedy with optimistic initialisation", "gradient bandit"]
    colors = ['red', 'blue', 'black', 'green']
    lineChart(X, Y, colors, labels)
