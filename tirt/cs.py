# coding=utf-8
import numpy as np
from tirt import BaseModel


def irt_consistency_score(model, theta=None):
    # irt评分一致性，检测乱答，随机作答的被试
    if not isinstance(model, BaseModel):
        raise ValueError('mode must be bayes_probit or bayes_logistic or ml_probit or ml_logistic')

    if theta is not None:
        prob_val = model.prob(theta)
    else:
        theta = model.solve
        prob_val = model.prob(theta)

    score = model.score

    c_ratio = 1.0 * (len(prob_val[score == 1][prob_val[score == 1] > 0.5]) +
                     len(prob_val[score == 0][prob_val[score == 0] < 0.5])) / len(prob_val)
    return c_ratio


def ctt_consistency_score(score):
    # ctt评分一致性，检测乱答，随机作答的被试

    if not isinstance(score, np.ndarray):
        raise ValueError('mode must be bayes_probit or bayes_logistic or ml_probit or ml_logistic')

    return np.std(score)
