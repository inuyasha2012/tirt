from tirt import SimFixedTirt

fixed_tirt = SimFixedTirt(subject_nums=100, trait_size=30, items_size_per_dim=10)
theta_list = fixed_tirt.sim()
score_list = fixed_tirt.scores

for i, theta in enumerate(theta_list):
    print score_list[i]
    print theta
