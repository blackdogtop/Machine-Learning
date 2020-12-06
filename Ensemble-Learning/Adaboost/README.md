# A Simple Adaboost Classification Implementation
## Adaboost的简单实现之三个臭皮匠顶个诸葛亮
本文主要讲解一个简单的Adaboost实现，对应代码可见 [A simple Adaboost implementation](https://github.com/blackdogtop/Machine-Learning/blob/main/Ensemble-Learning/Adaboost/adaboost.py). <br/>

主要数据和计算过程也可参考 [手把手教你实现一个 AdaBoost](https://developer.ibm.com/zh/technologies/analytics/articles/machine-learning-hands-on6-adaboost/) <br/>

此外，还可以参考[scikit-learn Adaboost类库使用小结](https://www.cnblogs.com/pinard/p/6136914.html) 和其对应的代码 [adaboost-classifier](https://github.com/ljpzzz/machinelearning/blob/master/ensemble-learning/adaboost-classifier.ipynb)。其调用sklearn库实现了一个基于决策树的Adaboost分类。

## Adaboost原理
Adaboost是一种Boosting算法，弱学习器与弱学习器之间存在强依赖关系。简单来说，Adaboost的实现流程是首先初始化样本权重，进入迭代并计算误差率与弱学习器的权重，根据弱学习器的表现对样本权重分布进行调整，使之前弱学习器做错的样本在后续受到更多关注，当达到最大迭代次数或强学习器误差率低于一定阈值停止迭代。<br/>

Adaboost的伪代码如下
![adaboost pseudo code](https://raw.githubusercontent.com/blackdogtop/image-host/master/Machine-Learning/Ensemble-Learning/Adaboost/Adaboost-pseudocode.png)

此外，更详细的Adaboost原理及解释可参考
**李航 - 统计学习方法
周志华 - 机器学习
[集成学习之Adaboost算法原理小结](https://www.cnblogs.com/pinard/p/6133937.html)**

## Adaboost的样例解释
本小结使用[手把手教你实现一个 AdaBoost](https://developer.ibm.com/zh/technologies/analytics/articles/machine-learning-hands-on6-adaboost/)中的样本数据，旨在实现一个简单的二分类adaboost，详细的计算过程或原理解释也可参考链接。
### 样本数据的创建
| x | 0 | 1 | 2  | 3  | 4 | 5  |
|---|---|---|----|----|---|----|
| y | 1 | 1 | -1 | -1 | 1 | -1 |

其中x表示样本数据，y表示对应每个样本的标签(1或-1)

### 初始化样本权重
每个样本权重的初始化可直接置为*1/N*，其中N为样本的个数。<br/>

故初始化得到的样本权重为 **(0.167, 0.167, 0.167, 0.167, 0.167, 0.167)** <br/>

初始化样本权重代码如下
```
self.weights = [1/len(x)] * len(x)  # init sample weights
```

### 可能的切分点
样本的切分点可选 (0.5, 1.5, 2.5, 3.5, 4.5) 中的一个，实际的选择应对比每个切分点的误差率，选择一个最优的弱学习器。
样本切分点的list创建代码如下
```
self.svList = [(x[i] + x[i - 1]) / 2 for i in range(1, len(x))]  # split value of samples
```

### 误差率计算
误差率的计算公式如下 <br/>
![error rate](https://raw.githubusercontent.com/blackdogtop/image-host/master/Machine-Learning/Ensemble-Learning/Adaboost/error%20rate.svg) <br/>
*其中 <br/>
m = 1,2,..,M 表示第 m 轮迭代 <br/>
i 表示第 i 个样本 <br/>
W 是样本权重 <br/>
I 表示函数取值为1或0，当括号中的表达式为真时，I 结果为1，反之为0* <br/>

如下表所示，根据误差率计算公式可以获取每个可能的切分点产生的弱学习器的误差率，并选择误差率最小的切分点作为最优弱学习器。在第一轮迭代中，当切分点为 1.5 时的误差率最小，**e = 0.167**

|     | hm(when x < split value) | hm(when x > split vaue) | 误差率            |
|-----|-------------------------|------------------------|-------------------|
| 0.5 | 1                       | -1                     | 2 * 0.167 = 0.334 |
| 1.5 | 1                       | -1                     | 1 * 0.167 = 0.167 |
| 2.5 | 1                       | -1                     | 2 * 0.167 = 0.334 |
| 3.5 | 1                       | -1                     | 3 * 0.167 = 0.501 |
| 4.5 | 1                       | -1                     | 2 * 0.167 = 0.334 |

误差率计算代码如下
```
def getErrorRate(self):
    """对比所有可能切分点的误差率并获取最小误差率"""
    def getSingleErrorRate(splitValue):
        """判断以split value 为切分点的两种方式里 误差率最小的一个"""
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
```

### 弱学习器权重计算
弱学习器权重计算公式如下 <br/>
![individual learner weight](https://raw.githubusercontent.com/blackdogtop/image-host/master/Machine-Learning/Ensemble-Learning/Adaboost/individual%20learner%20weight.svg) <br/>
*其中 m = 1,2,..,M 表示第 m 轮迭代 <br/>
可见当弱学习器误差率越小其权重越高，当弱学习器误差率越大其权重越小 <br/>
这样可以使分类精度高的弱学习器起到更大的作用，并削弱精度低的弱学习器的作用。* <br/>

经计算，在第一轮迭代中弱学习器的权重 **a = 0.5 * ln((1 – 0.167) / 0.167) = 0.8047** <br/>

弱学习器权重计算代码如下
```
def getAlpha(self, errorRate):
    alpha = 1/2 * math.log((1-errorRate) / errorRate)
    self.alphaList.append(alpha)
    return alpha
```

### 样本权重更新
样本权重更新公式如下 <br/>
![update sample weight](https://raw.githubusercontent.com/blackdogtop/image-host/master/Machine-Learning/Ensemble-Learning/Adaboost/update%20sample%20weight.svg) <br/>
*Z 为规范化因子，其计算公式如下* <br/>
![update sample weight - normalisation](https://raw.githubusercontent.com/blackdogtop/image-host/master/Machine-Learning/Ensemble-Learning/Adaboost/update%20sample%20weight%20-%20normalisation.svg) <br/>
*其中 <br/>
W 表示样本权重 <br/>
m = 1,2,..,M 表示第 m 轮迭代 <br/>
i 表示第 i 个样本 <br/>
am 表示第m轮迭代的弱学习器的权重 <br/>
当样本被正确分类，exp内为负值，新样本权重变小，反之exp内为正值，新样本权重变大 <br/>
可见当错误分类时，新的样本权重会变大，会在下一轮迭代中得到重视* <br/>

根据上述公式可得到在第一轮迭代中每个样本更新之后的样本权重 **(0.1, 0.1, 0.1, 0.1, 0.5, 0.1)**

|   | 分类结果 | 样本权重                     | 规范化               |
|---|----------|------------------------------|----------------------|
| 0 | 正确     | 0.167 * exp(-0.8047) = 0.075 | 0.075 / 0.748 = 0.10 |
| 1 | 正确     | 0.167 * exp(-0.8047) = 0.075 | 0.075 / 0.748 = 0.10 |
| 2 | 正确     | 0.167 * exp(-0.8047) = 0.075 | 0.075 / 0.748 = 0.10 |
| 3 | 正确     | 0.167 * exp(-0.8047) = 0.075 | 0.075 / 0.748 = 0.10 |
| 4 | 错误     | 0.167 * exp(0.8047) = 0.373  | 0.373 / 0.748 = 0.50 |
| 5 | 正确     | 0.167 * exp(-0.8047) = 0.075 | 0.075 / 0.748 = 0.10 |

样本权重更新代码如下
```
def updateWeights(self, alpha, prediction):
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
```

### 强学习器误差率
每一轮迭代完成后可以计算强学习器的预测计算其误差率，强学习器的公式如下 <br/>
![strong learner](https://raw.githubusercontent.com/blackdogtop/image-host/master/Machine-Learning/Ensemble-Learning/Adaboost/strong-learner.svg) <br/>
*其中 <br/>
G(x) 表示强学习器 <br/>
sign 表示一个非线性函数，当输入值大于 0 时 sign(input) 为 1，当输入值小于 0 时 sign(intput) 为 -1 <br/>
m 表示第 m 轮迭代 <br/>
a 表示弱学习器权重 <br/>
h 表示弱学习器* <br/>

故当第一轮迭代结束，此时的强学习器为 
G(x) = sign(alpha * h(x) ) 
= sign(0.8047 * [1, 1, -1, -1, -1, -1]) 
= [1, 1, -1, -1, -1, -1]
对于正确标签[1, 1, -1, -1, 1, -1]，此时的误差率为 1/6 = 0.167
如果此时达到了预设的误差率阈值或最大迭代次数则停止迭代，否则继续迭代。

### 三次迭代的结果
根据计算可以获得三次迭代的误差率，弱学习器权重，样本权重和强学习器错误率，结果如下表所示

| 迭代次数<div style="width: 45pt"> | 误差率 | 弱学习器权重<div style="width: 65pt"> | 样本权重<div style="width: 310pt">                                         | 强学习器错误率<div style="width: 80pt"> |
|----------|--------|--------------|--------------------------------------------------|----------------|
| 1        | 0.167  | 0.8047       | (0.1, 0.1, 0.1, 0.1, 0.5, 0.1)                   | 0.167          |
| 2        | 0.2    | 0.6931       | (0.0625, 0.0625, 0.250, 0.250, 0.3125, 0.0625)   | 0.167          |
| 3        | 0.1875 | 0.7332       | (0.1667, 0.1667, 0.1539, 0.1539, 0.1923, 0.1667) | 0              |


运行代码，将自动迭代三次，并输出每次迭代的结果如下所示。
```
In the 1 iteration:
误差率: 0.16666666666666666
弱学习器权重: 0.8047189562170503
样本权重: [0.1, 0.1, 0.1, 0.1, 0.5000000000000001, 0.1]
强学习器错误率: 0.16666666666666666

In the 2 iteration:
误差率: 0.2
弱学习器权重: 0.6931471805599453
样本权重: [0.0625, 0.0625, 0.25, 0.25, 0.31250000000000006, 0.0625]
强学习器错误率: 0.16666666666666666

In the 3 iteration:
误差率: 0.1875
弱学习器权重: 0.7331685343967135
样本权重: [0.16666666666666666, 0.16666666666666666, 0.15384615384615385, 0.15384615384615385, 0.19230769230769235, 0.16666666666666666]
强学习器错误率: 0.0

strong learner prediction: [1, 1, -1, -1, 1, -1]
```

### 完整代码
见[adaboost.py](https://github.com/blackdogtop/Machine-Learning/blob/main/Ensemble-Learning/Adaboost/adaboost.py)

## 参考及相关阅读
李航 - 统计学习方法 <br/>
周志华 - 机器学习 <br/>
[集成学习之Adaboost算法原理小结](https://www.cnblogs.com/pinard/p/6133937.html) <br/>
[手把手教你实现一个 AdaBoost](https://developer.ibm.com/zh/technologies/analytics/articles/machine-learning-hands-on6-adaboost/) <br/>
[AdaBoost](https://en.wikipedia.org/wiki/AdaBoost) <br/>
