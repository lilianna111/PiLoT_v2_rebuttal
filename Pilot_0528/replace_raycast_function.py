#!/usr/bin/env python3
"""
替换ray_casting.py中的caculate_predictXYZ_batch函数
"""
import re

# 读取ray_casting.py
with open('pixloc/crop/ray_casting.py', 'r') as f:
    original = f.read()

# 读取ray_casting_fix.py
with open('pixloc/crop/ray_casting_fix.py', 'r') as f:
    fix = f.read()

# 提取新函数的完整内容（包括函数签名）
new_func_match = re.search(r'(def caculate_predictXYZ_batch_enu\(.*?\n(?:.*?\n)*?    return result_ecef, k_value, result_sample_height)', fix, re.DOTALL)

if not new_func_match:
    print("错误: 未找到新函数")
    exit(1)

new_func = new_func_match.group(1)
# 修改函数名
new_func = new_func.replace('caculate_predictXYZ_batch_enu', 'caculate_predictXYZ_batch')

print(f"提取的新函数长度: {len(new_func)} 字符")

# 查找原函数位置
old_func_match = re.search(r'(    def caculate_predictXYZ_batch\(.*?\n(?:.*?\n)*?        return points, k_value, result_sample_height)', original, re.DOTALL)

if not old_func_match:
    print("错误: 未找到原函数")
    exit(1)

old_func = old_func_match.group(1)
print(f"找到原函数，长度: {len(old_func)} 字符")

# 替换
modified = original.replace(old_func, new_func)

# 写回文件
with open('pixloc/crop/ray_casting.py', 'w') as f:
    f.write(modified)

print("✓ 替换完成!")
print(f"原文件大小: {len(original)} → 新文件大小: {len(modified)}")
