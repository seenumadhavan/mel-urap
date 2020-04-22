from cnn import prep_val, train, val_dice, test

'''
# validation data prep
prep_val('f1', 'exp4', '/jewel-scratch')
prep_val('f2', 'exp4', '/jewel-scratch')
prep_val('f3', 'exp4', '/jewel-scratch')
prep_val('f4', 'exp4', '/jewel-scratch')
prep_val('f5', 'exp4', '/jewel-scratch')
'''

# training
# train('f1', 'exp4', '/jewel-scratch', 20, 39)
# train('f2', 'exp4', '/jewel-scratch', 33, 39)
# train('f3', 'exp4', '/jewel-scratch', 0, 39)
# train('f4', 'exp4', '/soge-home/projects/leaf-gpu', 28, 39)
# train('f5', 'exp4', '/soge-home/projects/leaf-gpu', 12, 39)

# validating
# val_dice('f1', 'exp4', '/jewel-scratch')
# val_dice('f2', 'exp4', '/jewel-scratch')
# val_dice('f3', 'exp4', '/soge-home/projects/leaf-gpu')
# val_dice('f4', 'exp4', '/soge-home/projects/leaf-gpu')
# val_dice('f5', 'exp4', '/soge-home/projects/leaf-gpu')

# testing
# test('f1', 'exp4', '/jewel-scratch', 0)
# test('f2', 'exp4', '/soge-home/projects/leaf-gpu', 5)
# test('f3', 'exp4', '/soge-home/projects/leaf-gpu', 0)
# test('f4', 'exp4', '/soge-home/projects/leaf-gpu', 0)
# test('f5', 'exp4', '/soge-home/projects/leaf-gpu', 0)
test('f6', 'exp1', '/soge-home/projects/leaf-gpu', 85)
