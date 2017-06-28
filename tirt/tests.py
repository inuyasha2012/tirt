import numpy as np
from tirt import BayesLogitModel, BayesProbitModel, MLProbitModel, MLLogitModel
from utils import sim_scores, random_params, gen_item_bank
from statsmodels import api as sm
from sim import SimAdaptiveTirt, SimFixedTirt
import multiprocessing

# print len(sim_scores(30, 10, 1)[0])

# item_dict = gen_item_dict(30, 10, block_size=2)
# bank_item_dict = gen_item_dict(30, 10)

# a, b = random_params(item_dict, 30, block_size=2)
# bank_a, bank_b = random_params(bank_item_dict, 30)

#
# print MLProbitModel(a, b, np.zeros(30), sim_scores(30, 10, 1)[0]).solve()
# scores = sim_scores(30, 10, 1)
#
# for score in scores:
#     m = BayesProbitModel(a, b, np.zeros(30), score)
#     t = m.newton()
#     print t
    # print BayesProbitModel(a, b, np.zeros(30), score).gradient_ascent()


def tt(num):
    SimAdaptiveTirt(item_size=600, subject_nums=num, trait_size=30, model='bayes_probit', block_size=2).sim()
#
if __name__ == '__main__':
    # multiprocessing.freeze_support()
    pool = multiprocessing.Pool(1)
    pool.map(tt, [100 for i in range(10)])

# res = SimFixedTirt(items_size_per_dim=5, trait_size=30, subject_nums=1000)
# res.sim()

# gen_item_bank(trait_size=5, item_size=200)

# print sm.GLM(scores[0], a, family=sm.families.Binomial()).fit()

# items = np.array(gen_item_bank(5, 200, 3))
# for _ in range(5):
#     print len(items[items == _])