# DataScience
数据科学方面学习和练习。

    假如大家的体重离均值比较大，也就是体重比较多样，就会导致TSS比较大；反之，大家的体重如果集中在均值附近，TSS就比较小。
　TSS实际上就是方差(variance),用来度量数据偏离均值的程度。数据为什么会偏离均值呢？因为个体之间(的自变量取值)是有差异的，导致它们的某个指标(因变量)各不相同。我们训练的多元线性回归模型是用来模拟自变量和因变量之间的关系的，那么，这个模型计算出来的因变量取值，与真实的取值之间，有多大的差距呢？我们可以构造一个指标,回归平方和(也叫可解释平方和，exlplained sum of suqres, ESS):
    ESS = (weight_pred_1 - weight_mean)^2 + (weight_pred_2 - weight_mean)^2+ ... + (weight_pred_N - weight_mean)^2
 　ESS度量的是模型计算出来的体重与真实的体重均值的偏离程度，实际上是模型里蕴含的关于年龄,身高和体重之间关系的知识的量的大小。
　TSS和ESS之间有什么关系呢?TSS ＝ ESS + RSS,式中RSS是残差平方和(residual sum of squares).而残差平方和的计算方法是:
    RSS = (weight_1 -weight_pred_1 )^2 + (weight_2 - weight_pred_2 )^2+ ... + (weight_N - weight_pred_N)^2
    残差平方和表示的是模型计算得到的因变量取值与真实值的差距，RSS越大，说明计算值与真实值差距越大。
我们构造一个指标，RSS/ESS,来表示残差带来的偏离，和模型带来的偏离，二者的对比。为什么要构造这个指标呢？暂时不知道原因，就假定是提出者凑出来的一个东西，它恰好具有一些性质，刚好可以解决某些问题。
　如果我们仔细看TSS-ESS-RSS的结构，实际上有一个差项。有很多方法可以证明这个差等于0(本文作者还没有证明过,为了学习进度，这种工作量大的环节就先欠着吧)。
　
2.模型整体是否显著？
这个问题的意思是，如果多元线性回归模型里的所有自变量的系数都为0、只剩截距的时候(weight=0+0+...+0+0)，模型是不是显著的？如果是显著的，说明我们只需要让weight等于一个恒定值，就可以对人的体重做出有效的预测了，也就是说，年龄、身高这些变量与体重没有关系。只有这个假设出来的精简模型不显著，说明我们的模型里至少有一个参数是不为0的、是有价值的，才有进一步考虑它的可用性的必要。这里的思想是统计学里经常用的，假设存在一个精简后的模型(0假设)，如果这个精简模型是显著的，说明我们构建的模型很可能对事物做出了过于复杂的假设和解释，需要仔细检查一下。
F检验的作用，就是做上面所说的显著性检验，简单来说。如果A=B=C=0时的模型(weight=0)，是显著的，反过来就说明我们训练得到的weight = A*age + B*height + C是没有意义的。
A=B=C=0时，
    ESS = (weight_pred_1 - weight_mean)^2 + (weight_pred_2 - weight_mean)^2+ ... + (weight_pred_N - weight_mean)^2
           =  weight_mean^2 + weight_mean^2+ ... + weight_mean^2
           = N*weight_mean^2

    RSS = (weight_1 -weight_pred_1 )^2 + (weight_2 - weight_pred_2 )^2+ ... + (weight_N - weight_pred_N)^2
           = (weight_1 )^2 + (weight_2 )^2+ ... + (weight_N)^2
           
由于weight_i服从一个正态分布(均值为weight_mean_real, 标准差为delta).基于方差的性质 D(X+Y)=DX+DY，以及D(kX)=k^2*DX，我们可以推导出weight_mean=(weight_1 + weight_2 + ... + weight_N)/N也服从一个正态分布(均值也是weight_mean_reall, 标准差是delta/N^0.5, 稍小一些)：
    方差D(weight_mean) = D(weight_1/N + weight_2/N + ...) = D(weight_1/N ) + D( weight_2/N ) + ...
                                =delta^2/N^2 + delta^2/N^2 + ...
                                 = delta^2/N
     标准差就是delta/N^0.5
也就是说 ESS里的N个weight_mean是独立同分布的正态随机变量;RSS里的N个weight_i也是。
独立同分布的正态随机变量的平方和服从卡方分布(也叫x方分布，开方分布)——ESS服从自由度为N-3的卡方分布 , RSS服从自由度为N-1的卡方分布。
两个卡方分布的商，服从f分布——我们要是知道了一个变量的分布，在知道这个量的取值的情况下，就可以计算出这个取值出现的概率，是大概率还是小概率。   
