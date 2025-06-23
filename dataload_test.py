from  datasets import load_dataset


"""
测试load_dataset加载得到数据集属性，以及数据类型
"""

import torch
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

#dataset=load_dataset("wikitext","wikitext-2-raw-v1",split='test') # 三个参数分别表示：数据集名称 数据集版本 数据集划分（train、validation、test）
testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test'  # 忽略分片大小校验
        )# 强制重新下载

print(dir(testdata))
print(type(testdata['text'])) #list

print(len(testdata['text'])) #一维数组，每个元素是一个str字符串