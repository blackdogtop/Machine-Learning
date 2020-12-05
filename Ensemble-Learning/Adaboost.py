#!usr/bin/env python
# -*- coding: utf-8 -*-
import math
import numpy as np
"""
Adaboost implementation
data sample can infer: https://developer.ibm.com/zh/technologies/analytics/articles/machine-learning-hands-on6-adaboost/
"""


class Adaboost:
    """a simple adaboost implementation (one-dimensional sample)"""
    def __init__(self, x=None, y=None):
        # init samples and labels
        if not x and not y:
            x = [0, 1, 2, 3, 4, 5]
            y = [1, 1, -1, -1, 1, -1]
        self.x = x
        self.y = y
        self.svList = [(x[i] + x[i - 1]) / 2 for i in range(1, len(x))]  # split value of samples

        self.weights = [1/len(x)] * len(x)  # init sample weights
        self.errThreshold = 0.1  # error threshold
        self.maxIterNum = 5  # max iteration

        self.hxList = []  # store prediction for each individual learner
        self.alphaList = []  # store weight (alpha) for each individual learner

    def getErrorRate(self):
        """
        calculate optimised individual learner according to split values
        :returns
            minErrorRate: the min error rate from different split values
            bestPrediction: a list stores predict result
        """
        def getSingleErrorRate(splitValue):
            """
            get min single split value error rate from two different situations
            two situations:
                1 when x > split value              -1 when x > split value
                -1 when x < split value     OR      1 when x < split
                error rate -- e                      error rate -- (1-e)
            :params splitValue: the single split value from cvList
            :returns
                min(ErrorRate): min error rate in two different situations
                prediction: a list stores predict result
            """
            errorRate = 0
            prediction = []  # store predict result
            for i, label in enumerate(self.y):
                predict = 1 if self.x[i] < splitValue else -1
                prediction.append(predict)
                if predict * label > 0:  # correct predict
                    errorRate += 0 * self.weights[i]
                else:  # wrong predict
                    errorRate += 1 * self.weights[i]
            if 1-errorRate < errorRate:
                errorRate = 1-errorRate
                prediction = [-1 * p for p in prediction]  # convert to additive inverse when min is the other situation
            return errorRate, prediction

        minErrorRate, bestPrediction = getSingleErrorRate(self.svList[0])  # initialise
        for i in range(1, len(self.svList)):
            errorRate, prediction = getSingleErrorRate(self.svList[i])
            if errorRate < minErrorRate:
                minErrorRate = errorRate
                bestPrediction = prediction
        self.hxList.append(bestPrediction)
        return minErrorRate

    def getAlpha(self, errorRate):
        """
        calculate individual learner weight(alpha), which is only related to error rate
        formula: 1/2 * log((1-errorRate) / errorRate)
        :params errorRate: error rate
        :return alpha: the individual learner weight
        """
        alpha = 1/2 * math.log((1-errorRate) / errorRate)
        self.alphaList.append(alpha)
        return alpha

    def updateWeights(self, alpha, prediction):
        """
        calculate and update sample weights
        flag (1 or -1) is used in the code to replace the result (yi * Gm(xi))
        formula:
            W(m+1, i) = W(m, i) / Z * exp(-alpha * yi * Gm(xi))
            Z = sum(exp(-alpha * yi * Gm(xi)))
            p.s. if individual learner Gm correct predict the sample xi, then the (-alpha * yi * Gm(xi)) is positive
                 vice versa
        :params
            alpha: current individual learner weight
            prediction: individual learner predict result
        """
        nextWeights = []  # new sample weights
        sumWeight = 0
        # get current total sample weights
        for i, weight in enumerate(self.weights):
            flag = 1 if prediction[i] == self.y[i] else -1  # (yi * Gm(xi)) is 1 if predict is correct else 0
            sumWeight = sumWeight + (self.weights[i] * math.exp(-1 * alpha * flag))
        # get new sample weights
        for i, weight in enumerate(self.weights):
            flag = 1 if prediction[i] == self.y[i] else -1
            nextWeight = self.weights[i] * math.exp(-1 * alpha * flag) / sumWeight
            nextWeights.append(nextWeight)
        self.weights = nextWeights  # update weights

    def sign(self, input):
        """
        sign function, which can convert input to 1 or -1
        :params input: input value
        :return output: 1 or -1
        """
        return 1 if input > 0 else -1

    def strongLearner(self):
        """
        get strong learner prediction
        formula:
            G(x) = sign(sum(a * h(x))
        :return gx: a list of strong learner prediction
        """
        # get error rate
        errorRate = self.getErrorRate()
        # get individual weight coefficient
        alpha = self.getAlpha(errorRate)
        # update data sample weights
        self.updateWeights(alpha, self.hxList[-1])  # hxList[-1] is the prediction of current individual learner

        # get strong learner output
        gx = np.array([0] * len(self.y))  # initialise
        for i, hx in enumerate(self.hxList):  # get prediction for each individual learner
            gx = gx + self.alphaList[i] * np.array(hx)
        gx = [self.sign(t) for t in gx]

        print('误差率: {}\n'
              '弱学习器权重: {}\n'
              '样本权重: {}'
              .format(errorRate, alpha, self.weights))

        return gx

    def trainAdaboost(self):
        """
        train adaboost and stop when the strong error rate less than error threshold or reach max iteration number
        :return gx: the output of strong learner
        """
        for iteration in range(self.maxIterNum):
            print('In the {} iteration:'.format(iteration + 1))
            errNum = 0
            gx = self.strongLearner()
            # calculate error rate
            for i in range(len(gx)):
                if gx[i] != self.y[i]:
                    errNum += 1
            errorRate = errNum / len(gx)

            print('强学习器错误率: {}\n'.format(errorRate))

            if errorRate < self.errThreshold:
                break
        return gx


if __name__ == '__main__':
    adaboost = Adaboost()
    gx = adaboost.trainAdaboost()
    print('strong learner prediction: {}'.format(gx))