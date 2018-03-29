# coding=utf-8
from __future__ import unicode_literals, print_function, absolute_import
from tirt import irt_consistency_score, sim_scores, BayesProbitModel, gen_item_dict, SimFixedTirt

model = SimFixedTirt(trait_size=30, items_size_per_dim=10, subject_nums=1000, model='bayes_probit')
model.sim()
for c_score in model.get_consistency_scores():
    print(c_score)

print('=======================================')

# 生成试题字典
item_dict = model._item_dict
# 生成试题参数
a, b = model.random_params
# 生成随机得分
scores = sim_scores(30, 10, 1000)

for score in scores:
    model = BayesProbitModel(a, b, score=score)
    # 打印一致性
    print(irt_consistency_score(model))



