import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange


class Bandit:
    def __init__(self, timeStep, arms=10, epsilon=0.1, alpha=0.1, sampleAverage=False):
        """
        :param arms:
        :param epsilon:
        :param timeStep:
        :param alpha: const step size
        :param sampleAverage: if true, use sample average to update new estimation instead of const step size
        """
        self.arms = arms
        self.epsilon = epsilon
        self.timeStep = timeStep
        self.alpha = alpha
        self.sampleAverage = sampleAverage
        self.q = np.zeros(arms)  # expected reward
        self.Q = np.zeros(arms)  # estimated reward
        self.N = [0 for _ in range(arms)]  # steps

    def reset(self):
        self.q = np.zeros(self.arms)
        self.Q = np.zeros(self.arms)
        self.N = [0 for _ in range(self.arms)]

    def execute(self):
        QinStep = np.zeros(shape=self.timeStep)
        optimalAction = np.zeros(shape=self.timeStep)

        for step in range(self.timeStep):
            randomInt = np.random.randint(0, 100)
            if randomInt < int(self.epsilon * 100):
                action = np.random.randint(0, self.arms)  # e-greedy method
            else:
                action = np.argmax(self.Q)  # greedy method
            reward = self.q[action] + np.random.randn()
            if self.sampleAverage:
                stepSize = (1. / self.N[action]) if self.N[action] else 0
            else:
                stepSize = self.alpha
            # Incremental computing
            self.Q[action] = self.Q[action] + stepSize * (reward - self.Q[action])
            self.N[action] += 1

            if action == np.argmax(self.q):
                optimalAction[step] = 1
            QinStep[step] = reward

            self.q = self.q + 0.01 * np.random.randn(self.arms)  # random walk
        return QinStep, optimalAction


def lineChart(X, Y, label):
    """plot line chart"""
    labels = ['average sampling', 'constant step-size']
    colors = ['r', 'b']
    for x, l, c in zip(X, labels, colors):
        plt.plot(Y, x, label=l, color=c)
    plt.xlabel('Steps')
    plt.ylabel(label)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    runs = 10000
    timeStep = 1000

    isSampleAverages = [True, False]
    averageReward = []
    optimalAction = []

    for i, isSampleAverage in enumerate(isSampleAverages):
        rewardsInRuns = []
        optimalActionInRuns = []
        for _ in trange(runs):
            bandit = Bandit(timeStep=timeStep, arms=10, epsilon=0.1, alpha=0.1, sampleAverage=isSampleAverage)
            rewards, actions = bandit.execute()
            rewardsInRuns.append(rewards)
            optimalActionInRuns.append(actions)
        rewardsInRuns = np.mean(np.array(rewardsInRuns), axis=0)
        optimalActionInRuns = np.mean(np.array(optimalActionInRuns), axis=0)

        averageReward.append(rewardsInRuns)  # average reward
        optimalAction.append(optimalActionInRuns)  # optimal action

    # plot in line chart
    lineChart(np.array(averageReward), range(timeStep), label="Average reward")
    lineChart(np.array(optimalAction), range(timeStep), label="Optimal action")
