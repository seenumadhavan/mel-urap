from cnn import prep_val, train, val_dice, test

# validation data prep
# prep_val('f1', 'exp3', '/scratch')
# prep_val('f2', 'exp3', '/scratch')
# prep_val('f3', 'exp3', '/scratch')
# prep_val('f4', 'exp3', '/scratch')
# prep_val('f5', 'exp3', '/scratch')

# training
# train('f1', 'exp3', '/scratch', 36, 39)
# train('f2', 'exp3', '/scratch', 12, 39)
# train('f3', 'exp3', '/scratch', 22, 39)
# train('f4', 'exp3', '/scratch', 6, 39)
# train('f5', 'exp3', '/scratch', 16, 39)

# validating
# val_dice('f1', 'exp3', '/scratch')
# val_dice('f2', 'exp3', '/soge-home/projects/leaf-gpu')
# val_dice('f3', 'exp3', '/soge-home/projects/leaf-gpu')
# val_dice('f4', 'exp3', '/soge-home/projects/leaf-gpu')
# val_dice('f5', 'exp3', '/soge-home/projects/leaf-gpu')

# testing
# test('f1', 'exp3', '/scratch', 0)
# test('f2', 'exp3', '/soge-home/projects/leaf-gpu', 45)
# test('f3', 'exp3', '/soge-home/projects/leaf-gpu', 0)
# test('f4', 'exp3', '/soge-home/projects/leaf-gpu', 0)
# test('f5', 'exp3', '/soge-home/projects/leaf-gpu', 0)
test('f6', 'exp1', '/soge-home/projects/leaf-gpu', 50)
