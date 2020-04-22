from cnn import prep_val, train, val_dice, test

'''
# validation data prep
prep_val('f1', 'exp6', '/jewel-scratch')
prep_val('f2', 'exp6', '/jewel-scratch')
prep_val('f3', 'exp6', '/jewel-scratch')
prep_val('f4', 'exp6', '/jewel-scratch')
prep_val('f5', 'exp6', '/jewel-scratch')
'''

# training
# train('f1', 'exp6', '/jewel-scratch', 35, 39)
# train('f2', 'exp6', '/jewel-scratch', 27, 39)
# train('f3', 'exp6', '/jewel-scratch', 24, 39)
# train('f4', 'exp6', '/jewel-scratch', 32, 39)
# train('f5', 'exp6', '/soge-home/projects/leaf-gpu', 11, 39)

# validating
# val_dice('f1', 'exp6', '/jewel-scratch')
# val_dice('f2', 'exp6', '/jewel-scratch')
# val_dice('f3', 'exp6', '/soge-home/projects/leaf-gpu')
# val_dice('f4', 'exp6', '/soge-home/projects/leaf-gpu')
# val_dice('f5', 'exp6', '/soge-home/projects/leaf-gpu')

# testing
# test('f1', 'exp6', '/jewel-scratch', 0)
# test('f2', 'exp6', '/soge-home/projects/leaf-gpu', 35)
# test('f3', 'exp6', '/soge-home/projects/leaf-gpu', 0)
# test('f4', 'exp6', '/soge-home/projects/leaf-gpu', 0)
# test('f5', 'exp6', '/soge-home/projects/leaf-gpu', 0)
test('f6', 'exp2', '/soge-home/projects/leaf-gpu', 85)