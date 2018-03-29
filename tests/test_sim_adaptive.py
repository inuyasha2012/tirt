from __future__ import unicode_literals, print_function, absolute_import
from tirt import SimAdaptiveTirt

sat = SimAdaptiveTirt(subject_nums=10, item_size=600, trait_size=30, max_sec_item_size=40, block_size=3)
sat.sim()
