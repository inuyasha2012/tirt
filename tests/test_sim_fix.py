from __future__ import unicode_literals, print_function, absolute_import
from tirt import SimFixedTirt

fixed_tirt = SimFixedTirt(subject_nums=1, trait_size=30, items_size_per_dim=2, block_size=2)
theta_list = fixed_tirt.sim()
score_list = fixed_tirt.scores

for i, theta in enumerate(theta_list):
    print(score_list[i])
    print(theta)
