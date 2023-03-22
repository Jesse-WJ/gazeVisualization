#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023-03-21 20:37
# @Author  : yunshang
# @FileName: create_model.py
# @Software: PyCharm
# @desc    : 


from .alexnet import get_model

def create_model(model_name='AlexNet'):
    if model_name == 'AlexNet':
        model = get_model()
    else:
        raise ValueError
    
    return model

if __name__ == '__main__':
    model = create_model()
    print(model)
