import multiprocessing
from tirt import SimAdaptiveTirt


def sim_adaptive(num):
    SimAdaptiveTirt(item_size=600, subject_nums=num, trait_size=30, model='bayes_probit', block_size=3).sim()

if __name__ == '__main__':
    pool = multiprocessing.Pool(5)
    pool.map(sim_adaptive, [100 for i in range(5)])