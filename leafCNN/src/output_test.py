import cnn
import os
from scipy import misc

val_file = misc.imread(os.path.join(".", "clahetina-3-2.png"))
prep_val('f_seq', 'exp_seq', 'scratch_dict', val_file)

# train('f1', 'exp1', '/scratch', 36, 39)
# train('f2', 'exp1', '/scratch', 12, 39)
# train('f3', 'exp1', '/scratch', 23, 39)
# train('f4', 'exp1', '/scratch', 6, 39)
# train('f5', 'exp1', '/scratch', 16, 39)
train('f6', 'exp1', '/soge-home/projects/leaf-gpu', 0, 39)

# validating
# val_dice('f1', 'exp1', '/scratch')
# val_dice('f2', 'exp1', '/soge-home/projects/leaf-gpu')
# val_dice('f3', 'exp1', '/soge-home/projects/leaf-gpu')
# val_dice('f4', 'exp1', '/soge-home/projects/leaf-gpu')
# val_dice('f5', 'exp1', '/soge-home/projects/leaf-gpu')
val_dice('f6', 'exp1', '/soge-home/projects/leaf-gpu')

# testing
# test('f1', 'exp1', '/scratch', 0)
# test('f2', 'exp1', '/soge-home/projects/leaf-gpu', 47)
# test('f3', 'exp1', '/soge-home/projects/leaf-gpu', 0)
# test('f4', 'exp1', '/soge-home/projects/leaf-gpu', 0)
# test('f5', 'exp1', '/soge-home/projects/leaf-gpu', 0)
test('f6', 'exp1', '/soge-home/projects/leaf-gpu', 0)