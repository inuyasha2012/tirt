# coding=utf-8
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