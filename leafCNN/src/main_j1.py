from cnn import prep_val, train, val_dice, test

# validation data prep
# prep_val('f1', 'exp1', '/scratch')
# prep_val('f2', 'exp1', '/scratch')
# prep_val('f3', 'exp1', '/scratch')
# prep_val('f4', 'exp1', '/scratch')
# prep_val('f5', 'exp1', '/scratch')
prep_val('f6', 'exp1', '/soge-home/projects/leaf-gpu')

# training
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