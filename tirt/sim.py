# coding=utf-8
import math
import numpy as np
from exception import UnknownModelError, IterMethodError
from utils import gen_item_dict, random_params, cached_property, gen_item_bank
from tirt import BayesProbitModel, MLProbitModel, MLLogitModel, BayesLogitModel
from cs import irt_consistency_score


class BaseSimTirt(object):

    MODEL = {'bayes_probit': BayesProbitModel, 'ml_probit': MLProbitModel,
             'bayes_logit': BayesLogitModel, 'ml_logit': MLLogitModel}

    def __init__(self, subject_nums, trait_size, model='bayes_probit',
                 iter_method='newton', block_size=3, lower=1, upper=4, avg=0, std=1):
        """

        :param subject_nums: int, 模拟被试的人数
        :param trait_size: int, 特质数量
        :param iter_method: str
        :param model: str, 模型
        :param block_size: int, 题块 
        :param lower: int|float
        :param upper: int|float
        :param avg: int|float
        :param std: int|float
        """
        if not isinstance(subject_nums, int):
            raise ValueError('subject_nums must be int')
        if not isinstance(trait_size, int):
            raise ValueError('trait_size must be int')
        if model not in ('bayes_probit', 'bayes_logistic', 'ml_probit', 'ml_logistic'):
            raise ValueError('mode must be bayes_probit or bayes_logistic or ml_probit or ml_logistic')
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
        if iter_method not in ('newton', 'gradient_ascent'):
            raise IterMethodError('iter_method must be newton or gradient_ascent')

        self._subject_nums = subject_nums
        self._trait_size = trait_size
        self._block_size = block_size
        self._lower = lower
        self._upper = upper
        self._avg = avg
        self.std = std
        self._iter_method = iter_method
        self._model = self._get_model(model)

    def _get_model(self, model):
        try:
            return self.MODEL[model]
        except KeyError:
            raise UnknownModelError('unknown model, must be "bayes_probit" or '
                                    '"ml_probit" or "bayes_logit" or "ml_logit"')

    @cached_property
    def random_thetas(self):
        """
        生成特质向量
        :return: ndarray
        """
        return np.random.multivariate_normal(np.zeros(self._trait_size),
                                             np.identity(self._trait_size), self._subject_nums)

    def _get_init_theta(self):
        return np.zeros(self._trait_size)

    def _get_mean_error(self, theta_list):
        return np.mean(np.abs(theta_list - self.random_thetas))


class SimFixedTirt(BaseSimTirt):

    # 模拟固定迫选测验

    def __init__(self, items_size_per_dim, *args, **kwargs):
        super(SimFixedTirt, self).__init__(*args, **kwargs)
        # 每个维度的题量
        self._items_size_per_dim = items_size_per_dim
        self.theta_list = []

    @cached_property
    def _item_dict(self):
        return gen_item_dict(self._trait_size, self._items_size_per_dim, self._block_size)

    @cached_property
    def random_params(self):
        """
        生成随机参数
        :return: tuple(ndarray, ndarray)
        """
        return random_params(self._item_dict, self._trait_size,
                             block_size=self._block_size, lower=self._lower,
                             upper=self._upper, avg=self._avg,
                             std=self.std)

    @cached_property
    def scores(self):
        """
        生成作答记录
        :return: list(ndarray)
        """
        slop, threshold = self.random_params
        thetas = self.random_thetas
        score_list = []
        for theta in thetas:
            p_list = self._model(slop, threshold).prob(theta)
            score = np.random.binomial(1, p_list, len(p_list))
            score_list.append(score)
        return score_list

    def sim(self):
        # 模拟作答
        slop, threshold = self.random_params
        init_theta = self._get_init_theta()
        theta_list = []
        for score in self.scores:
            # print score
            model = self._model(slop, threshold, init_theta, score, self._iter_method)
            try:
                res = model.solve
                theta_list.append(res)
            except Exception as e:
                theta_list.append(np.nan)
        mean_error = self._get_mean_error(np.array(theta_list))
        print '模拟结束，平均误差{0}'.format(mean_error)
        self.theta_list = theta_list
        return theta_list

    def get_consistency_scores(self):
        slop, threshold = self.random_params
        c_score_list = []
        for i, theta in enumerate(self.theta_list):
            model = self._model(slop, threshold, score=self.scores[i])
            c_score_list.append(irt_consistency_score(model, theta))
        return c_score_list


class SimAdaptiveTirt(BaseSimTirt):

    def __init__(self, item_size, max_sec_item_size=10, *args, **kwargs):
        super(SimAdaptiveTirt, self).__init__(*args, **kwargs)
        # 题库题量
        self._item_size = item_size
        # 已做答试题编号保存记录
        self._has_answered_item_idx = {}
        # 已做答得分保存记录
        self._score = {}
        # 参数估计保存记录
        self._theta = {}
        # 作答试题斜率保存记录
        self._slop = {}
        # 作答试题阈值保存记录
        self._threshold = {}
        # 第二阶段最大答题次数
        self._max_sec_item_size = max_sec_item_size

    @property
    def scores(self):
        return self._score

    @property
    def thetas(self):
        return self._theta

    def _add_slop(self, theta_idx, slop):
        if theta_idx in self._slop:
            self._slop[theta_idx] = np.concatenate((self._slop[theta_idx], slop))
        else:
            self._slop[theta_idx] = slop

    def _get_slop(self, theta_idx):
        return self._slop[theta_idx]

    def _get_threshold(self, theta_idx):
        return self._threshold[theta_idx]

    def _add_threshold(self, theta_idx, threshold):
        if theta_idx in self._threshold:
            self._threshold[theta_idx] = np.concatenate((self._threshold[theta_idx], threshold))
        else:
            self._threshold[theta_idx] = threshold

    def _add_answered_item_idx(self, theta_idx, used_item_idx_list):
        if theta_idx in self._has_answered_item_idx:
            self._has_answered_item_idx[theta_idx].extend(used_item_idx_list)
        else:
            self._has_answered_item_idx[theta_idx] = used_item_idx_list

    def _get_answered_item_idx_set(self, theta_idx):
        return set(self._has_answered_item_idx[theta_idx])

    def _get_can_use_items(self, theta_idx):
        can_use_idx = self._get_can_use_idx(theta_idx)
        return self.item_bank[list(can_use_idx)]

    def _get_can_use_idx(self, theta_idx):
        can_use_idx = self._item_idx_set - self._get_answered_item_idx_set(theta_idx)
        return can_use_idx

    def _add_score(self, theta_idx, score):
        if theta_idx in self._score:
            self._score[theta_idx] = np.concatenate((self._score[theta_idx], score))
        else:
            self._score[theta_idx] = score

    def _get_score(self, theta_idx):
        return self._score[theta_idx]

    def _add_theta(self, theta_idx, theta):
        if theta_idx in self._theta:
            self._theta[theta_idx].append(theta)
        else:
            self._theta[theta_idx] = [theta]

    def _get_theta(self, theta_idx):
        return self._theta[theta_idx][-1]

    @cached_property
    def item_bank(self):
        return gen_item_bank(self._trait_size, self._item_size, self._block_size)

    @cached_property
    def _item_idx_set(self):
        return set(range(self._item_size))

    def _get_random_choice_items(self, theta_idx):
        rand_choice_size = self._get_random_choice_size()
        while True:
            items = []
            dims = []
            used_idx_list = []
            idx_list = np.random.choice(list(self._item_idx_set), rand_choice_size, False)
            for i in idx_list:
                item = self.item_bank[i]
                items.append(item)
                dims.extend(item['dim'])
                used_idx_list.append(i)
            # if len(set(dims)) == self._trait_size:
            self._add_answered_item_idx(theta_idx, used_idx_list)
            return items

    def _get_random_choice_size(self):
        return int(math.ceil(1.0 * self._trait_size / self._block_size))

    def _get_random_choice_params(self, theta_idx):
        first_rand_items = self._get_random_choice_items(theta_idx)
        slop = []
        threshold = []
        for item in first_rand_items:
            slop.extend(item['params'][0])
            threshold.extend(item['params'][1])
        return np.array(slop), np.array(threshold)

    def _first_random(self, theta, theta_idx):
        # 第一阶段，随机抽题
        slop, threshold = self._get_random_choice_params(theta_idx)
        p_list = self._model(slop, threshold).prob(theta)
        score = np.random.binomial(1, p_list, len(p_list))
        init_theta = self._get_init_theta()
        model = self._model(slop, threshold, init_theta, score, self._iter_method)
        theta = model.solve
        self._add_score(theta_idx, score)
        self._add_theta(theta_idx, theta)
        self._add_slop(theta_idx, slop)
        self._add_threshold(theta_idx, threshold)

    def _second_random(self, theta, theta_idx):
        item = self._get_next_item(theta_idx)

        score = self._get_next_score(item, theta, theta_idx)
        # print score

        est_theta = self._get_estimate_theta(score, theta_idx)
        # print est_theta
        # print np.mean(np.abs(est_theta - theta))
        self._add_theta(theta_idx, est_theta)
        return est_theta

    def _get_estimate_theta(self, score, theta_idx):
        # 参数估计
        now_slop = self._get_slop(theta_idx)
        now_threshold = self._get_threshold(theta_idx)
        init_theta = self._get_init_theta()
        model = self._model(now_slop, now_threshold, init_theta, score, self._iter_method)
        est_theta = model.solve
        return est_theta

    def _get_next_score(self, item, theta, theta_idx):
        # 模拟自适应抽题的下一题得分
        item_slop = item['params'][0]
        self._add_slop(theta_idx, item_slop)
        item_threshold = item['params'][1]
        self._add_threshold(theta_idx, item_threshold)
        p_list = self._model(item_slop, item_threshold).prob(theta)
        item_score = np.random.binomial(1, p_list, len(p_list))
        self._add_score(theta_idx, item_score)
        score = self._get_score(theta_idx)
        return score

    def _get_next_item(self, theta_idx):
        # 获得自适应抽题的下一道题
        est_theta = self._get_theta(theta_idx)
        items = self._get_can_use_items(theta_idx)
        slop = self._get_slop(theta_idx)
        threshold = self._get_threshold(theta_idx)
        test_info = self._model(slop, threshold).info(est_theta)
        info_list = []
        for _item in items:
            _slop, _threshold = _item['params']
            item_info = self._model(_slop, _threshold).info(est_theta)
            info_list.append(np.linalg.det(test_info + item_info))
        max_info_idx = np.array(info_list).argmax()
        item = items[max_info_idx]
        idx = list(self._get_can_use_idx(theta_idx))[max_info_idx]
        self._add_answered_item_idx(theta_idx, [idx])
        return item

    def sim(self):
        thetas = self.random_thetas
        theta_list = []
        for i, theta in enumerate(thetas):
            try:
                est_theta = np.nan
                self._first_random(theta, i)
                for j in range(self._max_sec_item_size):
                    est_theta = self._second_random(theta, i)
                print u'第{0}个被试模拟成功！'.format(i + 1)
            except Exception as e:
                print e
                continue
            theta_list.append(est_theta)
        mean_error = self._get_mean_error(np.array(theta_list))
        print '模拟结束，平均误差{0}'.format(mean_error)
        return theta_list




