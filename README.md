# tirt
the simulation of Thurstone Item Response Theory, include fixed forced test and adaptive forced test. 模拟瑟斯顿项目反应理论，包括固定测验和自适应测验。

## 瑟斯顿IRT模型简介和应用
瑟斯顿IRT模型主要应用于迫选式非认知测验（人格测验，动机测验，兴趣测验等）。
瑟斯顿IRT模型同时也是一种多维项目反应理论（MIRT）模型。

###迫选式非认知测验
迫选测验形式可以如下

陈述 | 最符合 | 最不符合
----|------|----
一旦确定了目标，我会坚持努力地实现它| X| 
我是个勇于冒险，突破常规的人| | 
有我在的场合一般不会冷场| |X

也可以如下

陈述 | 最符合
----|------
一旦确定了目标，我会坚持努力地实现它| X 
我是个勇于冒险，突破常规的人|

也可以四个陈述一题，选最符合和最不符合或排序

当然最重要的一点是，这些陈述都是分属不同维度

## install
```
pip install tirt
```

##TIRT简介

###题型
支持三选二（一题三个陈述，选最符合和最不符合）和二选一（一题两个陈述，选最符合）

###模型
支持probit和logistic两种，如果你用的是mplus的WLSMV算法进行的项目参数估计，建议你使用probit模型

###参数估计
支持极大似然估计（ml）和贝叶斯极大后验（map）

###迭代算法
支持牛顿迭代和梯度上升，梯度上升更稳健，考虑加入更稳健的迭代加权最小二乘估计

## 固定测验模拟
模拟100个被试，30个维度，每个维度10个陈述，每道题3个陈述，所以模拟的测验总共有100题
```
from tirt import SimFixedTirt

fixed_tirt = SimFixedTirt(subject_nums=100, trait_size=30, items_size_per_dim=10)
theta_list = fixed_tirt.sim()
score_list = fixed_tirt.scores

for i, theta in enumerate(theta_list):
    print score_list[i]
    print theta
```

## 自适应测验模拟
模拟1个被试，题库600道题，30个维度，首先随机抽10题，第二阶段抽最合适的题，每次抽1道，终止规则是40题结束，总共50道题
```
from tirt import SimAdaptiveTirt

sat = SimAdaptiveTirt(subject_nums=1, item_size=600, trait_size=30, max_sec_item_size=40)
sat.sim()

for key, value in sat.thetas.items():
    print sat.scores[key]
    print value
```

自适应测验的模拟结果显示，自适应测验50题的精度，相当于固定测验90题的精度，自适应测验能减少44%的题量

测验类型|题量|平均误差
-------|-----|--------
自适应|50|0.24
固定|90|0.24

## 一致性
迫选测验通常都没有测谎量表（迫选测验本身抗作假），而衡量被试是否认真作答有更好的一致性分数
```
from tirt import irt_consistency_score, sim_scores, BayesProbitModel, gen_item_dict, SimFixedTirt
from tirt.utils import random_params

# 生成试题字典
item_dict = gen_item_dict(30, 10, block_size=3)
# 生成试题参数
a, b = random_params(item_dict, 30, block_size=3)
# 生成随机得分
scores = sim_scores(30, 10, 10)

for score in scores:
    model = BayesProbitModel(a, b, score=score)
    # 打印一致性
    print irt_consistency_score(model)

model = SimFixedTirt(trait_size=30, items_size_per_dim=10, subject_nums=100, model='bayes_probit')
model.sim()
print model.get_consistency_scores()
```

##API
详见源码注释