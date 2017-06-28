# coding=utf-8
import random
import numpy as np
from settings import TRIPLETS_PERMUTATION


def _gen_item_pattern(trait_size, items_size_per_dim, block_size=3):
    """
    生成迫选问卷试题排列组合，呈现形式为维度的排列
    :param trait_size: int, 特质数量
    :param items_size_per_dim: int, 每个特质的题量
    :return: list(int), 模拟正式试题的维度排列
    """
    item_pool = []
    for i in range(items_size_per_dim):
        item_pool.extend(range(trait_size))
    items = []
    while True:
        if not item_pool:
            break
        _dim_list = np.random.choice(item_pool, block_size)
        if len(set(_dim_list)) == block_size:
            # 每个题块的试题均要属于不同维度
            for dim in _dim_list:
                item_pool.remove(dim)
            items.extend(list(_dim_list))
    return items


def _gen_item_dt(items):
    """
    生成试题字典，key为试题顺序，value为试题所在维度
    :param items: list(int), 试题维度排列列表
    :return: dict(int: int) 试题字典
    """
    item_dt = {}
    for i, item in enumerate(items):
        item_dt[i] = item
    return item_dt


def gen_item_dict(trait_size, items_size_per_dim, block_size=3):
    items = _gen_item_pattern(trait_size, items_size_per_dim, block_size)
    item_dt = _gen_item_dt(items)
    print u'试题生成成功！'
    return item_dt


def gen_item_bank(trait_size, item_size, block_size=3, lower=1, upper=4, avg=0, std=1):
    """
    生成用于自适应测验的题库
    :param trait_size: int
    :param item_size: int
    :param block_size: int
    :param lower: int|float
    :param upper: int|float
    :param avg: int|float
    :param std: int|float
    :return: 
    """
    if not isinstance(trait_size, int):
        raise ValueError('trait_size must be int')
    if not isinstance(item_size, int):
        raise ValueError('trait_size must be int')
    if block_size not in (2, 3):
        raise ValueError('block_size must be 2 or 3')
    if not isinstance(lower, (int, float)):
        raise ValueError('lower must be int or float')
    if not isinstance(upper, (int, float)):
        raise ValueError('upper must be int or float')
    if not isinstance(avg, (int, float)):
        raise ValueError('avg must be int or float')
    if not isinstance(std, (int, float)):
        raise ValueError('std must be int or float')

    trait_list = range(trait_size)
    item_bank = []
    for i in range(item_size):
        _item_list = np.random.choice(trait_list, block_size, False)
        _item_dt = _gen_item_dt(_item_list)
        params = random_params(_item_dt, trait_size, block_size=block_size, lower=lower, upper=upper, avg=avg, std=std)
        item_bank.append({'dim': _item_list, 'params': params})
    return np.array(item_bank)


def _sim_score(block_nums):
    """
    生成随机试题得分（配对比较得分模式）
    :param block_nums: 题库的数量
    :return: 配对比较得分列表
    """
    score = []
    for j in range(block_nums):
        _score = random.choice(TRIPLETS_PERMUTATION)
        score.extend(_score)
    score = np.array(score)
    return score


def _get_block_nums(trait_size, items_size_per_dim, block_size=3):
    return trait_size * items_size_per_dim / block_size


def sim_scores(trait_size, items_size_per_dim, score_nums):
    """
    生成多个随机试题得分（配对比较得分模式）
    :param items_size_per_dim: int
    :param trait_size: int
    :param score_nums: int, 分数数量
    :return: list(ndarray), 多个得分列表
    """
    if not isinstance(trait_size, int):
        raise ValueError('trait_size must be int')
    if not isinstance(items_size_per_dim, int):
        raise ValueError('items_size_per_dim must be int')
    if not isinstance(score_nums, int):
        raise ValueError('score_nums must be int')

    block_nums = _get_block_nums(trait_size, items_size_per_dim)
    return [_sim_score(block_nums) for _ in range(score_nums)]


def _get_triplet_ipsative_score(pair_score, trait_size, item_dt, item_sign):
    """
    配对比较分数转自比分数
    :param pair_score: list(int), 配对比较分数
    :param trait_size: int, 特质的数量
    :param item_dt: dict(int: int), 试题维度字典
    :param item_sign: list(1|-1), 试题正反向列表
    :return: list(int), 自比分数列表
    """
    score = [0 for i in range(trait_size)]
    for j in range(len(pair_score) / 3):
        i1i2 = pair_score[j * 3]
        i1i3 = pair_score[j * 3 + 1]
        i2i3 = pair_score[j * 3 + 2]
        i1 = 1
        i2 = 0
        i3 = -1
        if i1i2 == 0:
            i1, i2 = i2, i1
        if i1i3 == 0:
            i1, i3 = i3, i1
        if i2i3 == 0:
            i2, i3 = i3, i2
        score[item_dt[j * 3]] += i1 * item_sign[item_dt[j * 3]]
        score[item_dt[j * 3 + 1]] += i2 * item_sign[item_dt[j * 3 + 1]]
        score[item_dt[j * 3 + 2]] += i3 * item_sign[item_dt[j * 3 + 2]]
    return score


def _pair_random_params(item_dt, trait_size, lower=1, upper=4, avg=0, std=1):
    """
    生成block_size为2的多维随机斜率（区分度）,和一维阈值（通俗度）
    :param std: 多维正态分布的标准差
    :param avg: 多维正态分布的期望值
    :param item_dt: dict(int:int), 试题字典
    :param trait_size: int，特质数量
    :param lower: int(>0), uniform分布的下界
    :param upper: int(>lower), uniform分布的上界
    :return: tuple(ndarray, ndarray), 斜率和阈值
    """
    keys = item_dt.keys()
    pair_nums = len(keys) / 2
    keys.sort()
    a = np.zeros((pair_nums, trait_size))
    a1 = np.random.uniform(lower, upper, pair_nums * 2)
    a2 = np.random.uniform(lower, upper, pair_nums * 2)

    for i in range(pair_nums):
        i1 = item_dt[2 * i]
        i2 = item_dt[2 * i + 1]
        a[i][i1] = a1[i]
        a[i][i2] = a2[2 * i] * -1
    b = np.random.normal(avg, std, pair_nums)
    return a, b


def _triplet_random_params(item_dt, trait_size, lower=1, upper=4, avg=0, std=1):
    """
    生成block_size为3的多维随机斜率（区分度）,和一维阈值（通俗度）
    :param std: 多维正态分布的标准差
    :param avg: 多维正态分布的期望值
    :param item_dt: dict(int:int), 试题字典
    :param trait_size: int，特质数量
    :param lower: int(>0), uniform分布的下界
    :param upper: int(>lower), uniform分布的上界
    :return: tuple(ndarray, ndarray), 斜率和阈值
    """
    keys = item_dt.keys()
    pair_nums = len(keys)
    keys.sort()
    a = np.zeros((pair_nums, trait_size))
    a1 = np.random.uniform(lower, upper, pair_nums)
    a2 = np.random.uniform(lower, upper, pair_nums)

    for i in range(len(keys) / 3):
        i1 = item_dt[3 * i]
        i2 = item_dt[3 * i + 1]
        i3 = item_dt[3 * i + 2]
        a[3 * i][i1] = a1[3 * i]
        a[3 * i][i2] = a2[3 * i] * -1
        a[3 * i + 1][i1] = a1[3 * i + 1]
        a[3 * i + 1][i3] = a2[3 * i + 1] * -1
        a[3 * i + 2][i2] = a1[3 * i + 2]
        a[3 * i + 2][i3] = a2[3 * i + 2] * -1
    b = np.random.normal(avg, std, pair_nums)
    return a, b


def random_params(item_dt, trait_size, block_size=3, lower=1, upper=4, avg=0, std=1):
    """
    生成随机参数
    :param item_dt: dict,试题字典，例如题块为3的0:1,1:0,2:2}代表第1题的第一个陈述测的是特质1，
    第二个陈述测的是特质0，第三个陈述测的是特质2
    :param trait_size: int
    :param block_size: int
    :param lower: int|float
    :param upper: int|float
    :param avg: int|float
    :param std: int|float
    :return: 
    """
    if not isinstance(item_dt, dict):
        raise ValueError('item_dt must be dict')
    if not isinstance(trait_size, int):
        raise ValueError('trait_size must be int')
    if block_size not in (2, 3):
        raise ValueError('block_size must be 2 or 3')
    if not isinstance(lower, (int, float)):
        raise ValueError('lower must be int or float')
    if not isinstance(upper, (int, float)):
        raise ValueError('upper must be int or float')
    if not isinstance(avg, (int, float)):
        raise ValueError('avg must be int or float')
    if not isinstance(std, (int, float)):
        raise ValueError('std must be int or float')

    if block_size == 3:
        return _triplet_random_params(item_dt, trait_size, lower=lower, upper=upper, avg=avg, std=std)
    elif block_size == 2:
        return _pair_random_params(item_dt, trait_size, lower=lower, upper=upper, avg=avg, std=std)


class cached_property(object):
    """
    # 从django抄的详见同名函数
    """
    def __init__(self, func, name=None):
        self.func = func
        self.__doc__ = getattr(func, '__doc__')
        self.name = name or func.__name__

    def __get__(self, instance, type=None):
        if instance is None:
            return self
        res = instance.__dict__[self.name] = self.func(instance)
        return res