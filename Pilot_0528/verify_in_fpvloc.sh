#!/bin/bash
# WGS84迁移验证脚本 - 使用FPVLoc环境
# 使用方法：bash verify_in_fpvloc.sh

echo "=========================================="
echo "🚀 WGS84坐标系迁移验证 - FPVLoc环境"
echo "=========================================="

# 进入FPVLoc环境
echo ""
echo "📦 激活FPVLoc conda环境..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate FPVLoc

if [ $? -ne 0 ]; then
    echo "❌ FPVLoc环境激活失败，请检查环境名称"
    exit 1
fi

echo "✅ FPVLoc环境已激活"

# 进入项目目录
cd /home/amax/Documents/code/pilot_v2/pilot_v2/Pilot_0528

# 验证1：检查DSM文件是否存在
echo ""
echo "=========================================="
echo "📁 步骤1：检查DSM/DOM文件"
echo "=========================================="

DSM_WGS84="/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.3/0.3/wgs84dsm.tif"
DOM_WGS84="/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.3/0.3/wgs84dom.tif"

if [ -f "$DSM_WGS84" ]; then
    echo "✅ WGS84 DSM存在: $DSM_WGS84"
    gdalinfo "$DSM_WGS84" | grep -E "Size|Origin|Pixel|Coordinate System"
else
    echo "❌ WGS84 DSM不存在: $DSM_WGS84"
    echo "   请确保已准备好WGS84坐标系的DSM文件"
    exit 1
fi

if [ -f "$DOM_WGS84" ]; then
    echo "✅ WGS84 DOM存在: $DOM_WGS84"
else
    echo "⚠️  WGS84 DOM不存在: $DOM_WGS84"
fi

# 验证2：运行Python验证脚本
echo ""
echo "=========================================="
echo "🔍 步骤2：运行反投影验证"
echo "=========================================="

python verify_wgs84_migration.py

VERIFY_RESULT=$?

# 验证3：运行一帧测试
echo ""
echo "=========================================="
echo "🎯 步骤3：运行单帧定位测试"
echo "=========================================="

echo "运行main.py测试（单帧）..."
# 这里可以添加main.py的测试命令
# python main.py -c configs/feicuiwan_m4t.yaml --name test_wgs84_single_frame

echo ""
echo "=========================================="
echo "📊 验证总结"
echo "=========================================="

if [ $VERIFY_RESULT -eq 0 ]; then
    echo "✅ WGS84迁移验证通过！"
    echo ""
    echo "下一步："
    echo "  1. 运行完整定位流程测试"
    echo "  2. 对比EPSG:4547和EPSG:4326的精度"
    echo "  3. 测量crop和定位时间"
else
    echo "⚠️  验证未通过，请检查错误信息"
fi

echo ""
echo "=========================================="
