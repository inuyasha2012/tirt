# coding=utf-8

"""
tirt
====

the simulation of Thurstone Item Response Theory, include fixed forced
test and adaptive forced test.
模拟瑟斯顿项目反应理论，包括固定测验和自适应测验。

瑟斯顿IRT模型简介和应用
-----------------------

瑟斯顿IRT模型主要应用于迫选式非认知测验（人格测验，动机测验，兴趣测验等）。

固定测验模拟
------------

模拟100个被试，30个维度，每个维度10个陈述，每道题3个陈述，所以下面这个陈述总共有100题

::

    from tirt import SimFixedTirt

    fixed_tirt = SimFixedTirt(subject_nums=100, trait_size=30, items_size_per_dim=10)
    theta_list = fixed_tirt.sim()
    score_list = fixed_tirt.scores

    for i, theta in enumerate(theta_list):
        print score_list[i]
        print theta

自适应测验模拟
--------------

模拟1个被试，题库600道题，30个维度，首先随机抽10题，第二阶段抽合适的题40道题，总共50道题

::

    from tirt import SimAdaptiveTirt

    sat = SimAdaptiveTirt(subject_nums=1, item_size=600, trait_size=30, max_sec_item_size=40)
    sat.sim()

    for key, value in sat.thetas.items():
        print sat.scores[key]
        print value

一致性
------

迫选测验通常都没有测谎量表（迫选测验本身抗作假），而衡量被试是否认真作答有更好的一致性分数

::

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
"""


from setuptools import setup

setup(
    name='tirt',
    version='0.0.4',
    packages=['tirt'],
    url='https://github.com/inuyasha2012/tirt',
    license='MIT',
    author='inuyasha2012',
    author_email='inuyasha021@163.com',
    description='the simulation of Thurstone Item Response Theory, include fixed forced test and adaptive forced test. ',
    long_description=__doc__,
    install_requires=['numpy', 'scipy'],
    classifiers=[
        'Programming Language :: Python :: 2.7',
    ]
)
