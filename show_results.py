#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_analysis_summary():
    """
    显示分析结果摘要
    """
    print("=" * 80)
    print("                    异常进程分析结果展示")
    print("=" * 80)
    
    # 读取分析结果
    result_path = '/mnt/ymj/vivo/异常进程/result/异常进程分析结果.csv'
    if os.path.exists(result_path):
        df = pd.read_csv(result_path)
        print(f"\n📊 分析统计:")
        print(f"   • 总异常进程数量: {len(df)}")
        print(f"   • 分析完成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 统计异常特征
        has_anomaly = df[df['异常特征'] != '未检测到明显异常特征']
        print(f"   • 检测到明显异常的进程: {len(has_anomaly)} ({len(has_anomaly)/len(df)*100:.1f}%)")
        
        # 统计各种异常类型
        anomaly_types = {
            '前台运行时间过长': 0,
            '后台运行时间过长': 0,
            '前台能耗过高': 0,
            '后台能耗过高': 0,
            '前台单位时间能耗过高': 0,
            '后台单位时间能耗过高': 0,
            '总单位时间能耗过高': 0,
            '设备电量较低': 0
        }
        
        for _, row in df.iterrows():
            features = row['异常特征']
            for anomaly_type in anomaly_types.keys():
                if anomaly_type in features:
                    anomaly_types[anomaly_type] += 1
        
        print(f"\n🔍 异常类型分布:")
        for anomaly_type, count in anomaly_types.items():
            if count > 0:
                print(f"   • {anomaly_type}: {count} ({count/len(df)*100:.1f}%)")
        
        # 显示最异常的5个进程
        print(f"\n⚠️  异常程度最高的5个进程:")
        top_anomalies = df.nsmallest(5, '异常分数')
        for i, (_, row) in enumerate(top_anomalies.iterrows(), 1):
            print(f"   {i}. {row['进程名称']}")
            print(f"      异常分数: {row['异常分数']:.4f}")
            print(f"      主要异常: {row['异常特征'][:100]}{'...' if len(row['异常特征']) > 100 else ''}")
            print()
        
        # 能耗统计
        print(f"📱 能耗分析:")
        avg_fg_energy = df['前台能耗_mAh'].mean()
        avg_bg_energy = df['后台能耗_mAh'].mean()
        avg_fg_energy_per_hour = df['前台单位时间能耗_mAh每小时'].mean()
        avg_bg_energy_per_hour = df['后台单位时间能耗_mAh每小时'].mean()
        
        print(f"   • 平均前台能耗: {avg_fg_energy:.2f} mAh")
        print(f"   • 平均后台能耗: {avg_bg_energy:.2f} mAh")
        print(f"   • 平均前台单位时间能耗: {avg_fg_energy_per_hour:.2f} mAh/小时")
        print(f"   • 平均后台单位时间能耗: {avg_bg_energy_per_hour:.2f} mAh/小时")
        
    else:
        print("❌ 未找到分析结果文件，请先运行分析程序")

def show_charts_info():
    """
    显示生成的图表信息
    """
    chart_dir = '/mnt/ymj/vivo/异常进程/result/charts'
    if os.path.exists(chart_dir):
        print(f"\n📈 生成的可视化图表:")
        charts = [
            ('anomaly_analysis_overview.png', '异常分析总览图'),
            ('anomaly_features_analysis.png', '异常特征分布图'),
            ('energy_efficiency_heatmap.png', '能耗效率相关性热力图')
        ]
        
        for chart_file, description in charts:
            chart_path = os.path.join(chart_dir, chart_file)
            if os.path.exists(chart_path):
                file_size = os.path.getsize(chart_path) / 1024  # KB
                print(f"   ✅ {description}")
                print(f"      文件: {chart_file} ({file_size:.1f} KB)")
                print(f"      路径: {chart_path}")
            else:
                print(f"   ❌ {description} - 文件未找到")
        
        print(f"\n💡 查看图表建议:")
        print(f"   • 可以使用图片查看器打开PNG文件")
        print(f"   • 建议按顺序查看: 总览图 → 特征分布图 → 相关性热力图")
        print(f"   • 图表展示了异常进程的分布特征和能耗模式")
    else:
        print(f"\n❌ 图表目录不存在: {chart_dir}")

def show_sample_analysis():
    """
    显示样例分析报告
    """
    json_path = '/mnt/ymj/vivo/异常进程/result/异常进程分析结果.json'
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if data:
            print(f"\n📋 样例分析报告 (第1个异常进程):")
            print("=" * 60)
            sample = data[0]
            print(f"进程名称: {sample['进程名称']}")
            print(f"设备型号: {sample['设备型号']}")
            print(f"异常分数: {sample['异常分数']:.4f}")
            print(f"异常特征: {sample['异常特征']}")
            print("\n详细分析报告:")
            print("-" * 40)
            # 只显示分析报告的前500个字符
            analysis = sample['分析结果']
            if len(analysis) > 500:
                print(analysis[:500] + "\n...（报告内容较长，完整内容请查看CSV或JSON文件）")
            else:
                print(analysis)
    else:
        print(f"\n❌ JSON结果文件不存在: {json_path}")

def show_file_locations():
    """
    显示所有生成文件的位置
    """
    print(f"\n📁 生成的文件位置:")
    files = [
        ('/mnt/ymj/vivo/异常进程/result/异常进程分析结果.csv', 'CSV格式分析结果'),
        ('/mnt/ymj/vivo/异常进程/result/异常进程分析结果.json', 'JSON格式分析结果'),
        ('/mnt/ymj/vivo/异常进程/result/charts/anomaly_analysis_overview.png', '异常分析总览图'),
        ('/mnt/ymj/vivo/异常进程/result/charts/anomaly_features_analysis.png', '异常特征分布图'),
        ('/mnt/ymj/vivo/异常进程/result/charts/energy_efficiency_heatmap.png', '能耗效率热力图'),
        ('/mnt/ymj/vivo/异常进程/data/anomaly_processes.csv', '原始异常进程数据'),
        ('/mnt/ymj/vivo/异常进程/data/all_process_data.csv', '完整进程数据集'),
        ('/mnt/ymj/vivo/异常进程/README.md', '项目说明文档')
    ]
    
    for file_path, description in files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / 1024  # KB
            print(f"   ✅ {description}")
            print(f"      {file_path} ({file_size:.1f} KB)")
        else:
            print(f"   ❌ {description} - 文件不存在")
            print(f"      {file_path}")

def main():
    """
    主函数
    """
    show_analysis_summary()
    show_charts_info()
    show_sample_analysis()
    show_file_locations()
    
    print("\n" + "=" * 80)
    print("🎉 异常进程分析和可视化完成！")
    print("\n📖 使用说明:")
    print("   1. 查看CSV文件获取完整的分析结果")
    print("   2. 查看PNG图表了解异常分布和特征")
    print("   3. 查看JSON文件获取结构化的分析数据")
    print("   4. 参考README.md了解项目详情")
    print("\n🔧 技术特点:")
    print("   • 基于Isolation Forest的异常检测算法")
    print("   • 考虑了单位时间能耗的创新指标")
    print("   • 区分游戏和服务应用的不同能耗模式")
    print("   • 提供详细的异常特征分析和优化建议")
    print("=" * 80)

if __name__ == "__main__":
    main()