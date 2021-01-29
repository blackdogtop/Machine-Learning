# -*- coding: utf-8 -*-
import numpy as np


class Bandit:
    """non-stationary bandit algorithms"""
    def __init__(self, timeStep, arms=10, alpha=0.1, sampleAverage=False, epsilon=0.1, c=2, optValue=5, baseLine=True):
        """
        :param timeStep:
        :param arms:
        :param alpha: step size
        :param sampleAverage: if true, use sample average to update new estimation instead of const step size
        :param epsilon: for e-greedy method
        :param c: for UCB
        :param optValue: for Optimistic Initial Values
        :param baseLine: far Gradient Bandit Algorithms
        """
        self.timeStep = timeStep
        self.arms = arms
        self.alpha = alpha
        self.sampleAverage = sampleAverage

        self.epsilon = epsilon
        self.c = c
        self.optValue = optValue
        self.baseLine = baseLine

        # init bandit parameters
        self.q = np.zeros(self.arms)
        self.Q = np.zeros(self.arms)
        self.N = np.zeros(self.arms)

    def epsilonGreedy(self):
        """
        epsilon-greedy method
        formula:
                    argmax Q(a)         with probability (1 - epsilon) (breaking ties randomly)
                A =
                    a random action     with probability epsilon
        :return: the average reward over the time step
        """
        averageRewardOverTimeStep = 0
        for step in range(self.timeStep):
            # e/greedy method
            randomInt = np.random.randint(0, 100)
            if randomInt < int(self.epsilon * 100):
                action = np.random.randint(0, self.arms)
            else:
                action = np.argmax(self.Q)
            # get reward
            reward = self.q[action] + np.random.randn()
            # update estimated reward
            if self.sampleAverage:
                stepSize = (1 / self.N[action]) if self.N[action] else 0
            else:
                stepSize = self.alpha
            self.Q[action] = self.Q[action] + stepSize * (reward - self.Q[action])  # incremental computing
            # update action step
            self.N[action] += 1
            # random walk
            self.q = self.q + 0.01 * np.random.randn(self.arms)
            # calculate average reward over time step
            averageRewardOverTimeStep = averageRewardOverTimeStep + (reward - averageRewardOverTimeStep) / (step + 1)
        return averageRewardOverTimeStep

    def UCB(self):
        """
        Upper-Confidence-Bound Action Selection
        formula:
                At = argmax[Qt(a) + c * sqrt(ln(t) / Nt(a))]
        :return: the average reward over the time step
        """
        averageRewardOverTimeStep = 0

        # init confidence
        At = np.zeros(self.arms)
        for step in range(self.timeStep):
            # update confidence
            for action in range(self.arms):
                if self.N[action] == 0:
                    At[action] = np.inf
                else:
                    At[action] = self.Q[action] + self.c * np.sqrt(np.log(step) / self.N[action])
            # action selection
            action = np.argmax(At)
            # get reward
            reward = self.q[action] + np.random.randn()
            # update estimated reward
            if self.sampleAverage: stepSize = (1 / self.N[action]) if self.N[action] else 0
            else: stepSize = self.alpha
            self.Q[action] = self.Q[action] + stepSize * (reward - self.Q[action])  # incremental computing
            # update action step
            self.N[action] += 1
            # random walk
            self.q = self.q + 0.01 * np.random.randn(self.arms)
            # calculate average reward over time step
            averageRewardOverTimeStep = averageRewardOverTimeStep + (reward - averageRewardOverTimeStep) / (step + 1)
        return averageRewardOverTimeStep

    def optimisticInitValues(self):
        """
        Optimistic Initial Values
        :return: the average reward over the time step
        """
        averageRewardOverTimeStep = 0

        # reset estimated reward to optimistic initial value
        for action in range(self.arms): self.Q[action] = self.optValue
        for step in range(self.timeStep):
            # greedy method
            action = np.argmax(self.Q)
            # get reward
            reward = self.q[action] + np.random.randn()
            # update estimated reward
            if self.sampleAverage: stepSize = (1 / self.N[action]) if self.N[action] else 0
            else: stepSize = self.alpha
            self.Q[action] = self.Q[action] + stepSize * (reward - self.Q[action])  # incremental computing
            # update action step
            self.N[action] += 1
            # random walk
            self.q = self.q + 0.01 * np.random.randn(self.arms)
            # calculate average reward over time step
            averageRewardOverTimeStep = averageRewardOverTimeStep + (reward - averageRewardOverTimeStep) / (step + 1)
        return averageRewardOverTimeStep

    def gradientBandit(self):
        """
        Gradient Bandit Algorithms
        :return: the average reward over the time step
        """
        averageRewardOverTimeStep = 0

        # init preference
        H = np.zeros(self.arms)
        for step in range(self.timeStep):
            # select action according to softmax
            PI = np.exp(H)
            PI = PI / np.sum(PI)
            action = np.argmax(PI)
            # get reward
            reward = self.q[action] + np.random.randn()
            # calculate average reward over time step
            averageRewardOverTimeStep = averageRewardOverTimeStep + (reward - averageRewardOverTimeStep) / (step + 1)
            # init baseline
            baseLine = averageRewardOverTimeStep if self.baseLine else 0
            # update preference
            for actionIndex in range(len(H)):
                predicate = 1 if action == actionIndex else 0
                H[actionIndex] = H[actionIndex] + self.alpha * (reward - baseLine) * (predicate - PI[actionIndex])
            # update action step
            self.N[action] += 1
            # random walk
            self.q = self.q + 0.01 * np.random.randn(self.arms)
        return averageRewardOverTimeStep
