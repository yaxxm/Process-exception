#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime

# 设置中文字体
import matplotlib.font_manager as fm

# 使用用户提供的字体文件
font_path = '/tmp/SourceHanSansSC-Regular.otf'
try:
    # 添加字体到matplotlib
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams['font.sans-serif'] = [font_prop.get_name(), 'SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
except:
    # 如果字体文件不存在，使用备用字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
    
plt.rcParams['axes.unicode_minus'] = False


def analyze_process_anomaly(row):
    """
    分析单个进程的异常特征
    """
    analysis = {
        'high_foreground_duration': row['foreground_duration'] > 3600,  # 前台时间超过1小时
        'high_background_duration': row['background_duration'] > 14400,  # 后台时间超过4小时
        'high_foreground_energy': row['foreground_energy'] > 500,  # 前台能耗过高
        'high_background_energy': row['background_energy'] > 200,  # 后台能耗过高
        'high_foreground_energy_per_hour': row['foreground_energy_per_hour'] > 150,  # 前台单位时间能耗过高
        'high_background_energy_per_hour': row['background_energy_per_hour'] > 100,  # 后台单位时间能耗过高
        'high_total_energy_per_hour': row['total_energy_per_hour'] > 200,  # 总单位时间能耗过高
        'low_battery': row['charge'] < 50,  # 电量过低
        'anomaly_severity': abs(row['anomaly_score'])  # 异常严重程度
    }
    return analysis

def generate_anomaly_description(row, anomaly_features):
    """
    生成异常描述
    """
    anomaly_desc = []
    if anomaly_features['high_foreground_duration']:
        anomaly_desc.append(f"前台运行时间过长({row['foreground_duration']/3600:.2f}小时)")
    if anomaly_features['high_background_duration']:
        anomaly_desc.append(f"后台运行时间过长({row['background_duration']/3600:.2f}小时)")
    if anomaly_features['high_foreground_energy']:
        anomaly_desc.append(f"前台能耗过高({row['foreground_energy']}mAh)")
    if anomaly_features['high_background_energy']:
        anomaly_desc.append(f"后台能耗过高({row['background_energy']}mAh)")
    if anomaly_features['high_foreground_energy_per_hour']:
        anomaly_desc.append(f"前台单位时间能耗过高({row['foreground_energy_per_hour']:.2f}mAh/小时)")
    if anomaly_features['high_background_energy_per_hour']:
        anomaly_desc.append(f"后台单位时间能耗过高({row['background_energy_per_hour']:.2f}mAh/小时)")
    if anomaly_features['high_total_energy_per_hour']:
        anomaly_desc.append(f"总单位时间能耗过高({row['total_energy_per_hour']:.2f}mAh/小时)")
    if anomaly_features['low_battery']:
        anomaly_desc.append(f"设备电量较低({row['charge']}%)")
    
    return "、".join(anomaly_desc) if anomaly_desc else "未检测到明显异常特征"

def generate_analysis_report(row, anomaly_features, anomaly_description):
    """
    生成分析报告（模拟大模型输出）
    """
    report = f"""## 异常进程分析报告

### 1. 进程基本信息分析
进程名称：{row['name']}
设备型号：{row['product']}
系统版本：{row['sysversion']}
运行时段：{row['formatted_start_time']} 至 {row['formatted_end_time']}

### 2. 运行时间异常分析
前台运行时长：{row['foreground_duration']/60:.1f}分钟
后台运行时长：{row['background_duration']/60:.1f}分钟
"""
    
    if anomaly_features['high_foreground_duration']:
        report += "⚠️ 前台运行时间异常：超过正常阈值，可能影响用户体验和电池续航。\n"
    if anomaly_features['high_background_duration']:
        report += "⚠️ 后台运行时间异常：长时间后台运行可能导致不必要的资源消耗。\n"
    
    report += f"""
### 3. 能耗异常分析
前台能耗：{row['foreground_energy']}mAh
后台能耗：{row['background_energy']}mAh
前台单位时间能耗：{row['foreground_energy_per_hour']:.2f}mAh/小时
后台单位时间能耗：{row['background_energy_per_hour']:.2f}mAh/小时
总单位时间能耗：{row['total_energy_per_hour']:.2f}mAh/小时
"""
    
    if anomaly_features['high_foreground_energy'] or anomaly_features['high_background_energy']:
        report += "⚠️ 能耗异常：总能耗超过正常范围，建议检查应用行为。\n"
    if anomaly_features['high_total_energy_per_hour']:
        report += "⚠️ 单位时间能耗异常：能耗效率低下，可能存在性能问题。\n"
    
    report += f"""
### 4. 设备状态分析
设备电量：{row['charge']}%
屏幕状态：{'开启' if row['screen'] == 1 else '关闭'}
"""
    
    if anomaly_features['low_battery']:
        report += "⚠️ 电量状态：设备电量较低，高能耗应用可能加速电量消耗。\n"
    
    report += f"""
### 5. 异常原因总结
异常分数：{row['anomaly_score']:.4f} (分数越低越异常)
检测到的异常特征：{anomaly_description}

### 6. 优化建议
"""
    
    if anomaly_features['high_foreground_duration']:
        report += "- 建议优化前台运行逻辑，减少不必要的前台时间\n"
    if anomaly_features['high_background_duration']:
        report += "- 建议检查后台任务，避免长时间后台运行\n"
    if anomaly_features['high_total_energy_per_hour']:
        report += "- 建议优化算法和资源使用，提高能耗效率\n"
    if anomaly_features['low_battery']:
        report += "- 建议在低电量时限制高能耗操作\n"
    
    if not any(anomaly_features.values()):
        report += "- 该进程虽被标记为异常，但各项指标相对正常，建议进一步监控\n"
    
    return report

def create_visualizations(data, output_dir):
    """
    创建可视化图表
    """
    # 创建图表目录
    chart_dir = os.path.join(output_dir, 'charts')
    os.makedirs(chart_dir, exist_ok=True)
    
    # 1. 异常分数分布图
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.hist(data['anomaly_score'], bins=20, alpha=0.7, color='red', edgecolor='black')
    plt.title('Anomaly Score Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # 2. 能耗分布图
    plt.subplot(2, 2, 2)
    plt.scatter(data['foreground_energy'], data['background_energy'], 
               c=data['anomaly_score'], cmap='viridis', alpha=0.6)
    plt.colorbar(label='Anomaly Score')
    plt.title('Energy Consumption Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Foreground Energy (mAh)')
    plt.ylabel('Background Energy (mAh)')
    plt.grid(True, alpha=0.3)
    
    # 3. 单位时间能耗分布
    plt.subplot(2, 2, 3)
    plt.scatter(data['foreground_energy_per_hour'], data['background_energy_per_hour'], 
               c=data['anomaly_score'], cmap='plasma', alpha=0.6)
    plt.colorbar(label='Anomaly Score')
    plt.title('Energy Per Hour Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Foreground Energy/Hour (mAh/h)')
    plt.ylabel('Background Energy/Hour (mAh/h)')
    plt.grid(True, alpha=0.3)
    
    # 4. 运行时间vs能耗
    plt.subplot(2, 2, 4)
    total_duration = data['foreground_duration'] + data['background_duration']
    total_energy = data['foreground_energy'] + data['background_energy']
    plt.scatter(total_duration/3600, total_energy, 
               c=data['anomaly_score'], cmap='coolwarm', alpha=0.6)
    plt.colorbar(label='Anomaly Score')
    plt.title('Duration vs Energy Consumption', fontsize=14, fontweight='bold')
    plt.xlabel('Total Duration (hours)')
    plt.ylabel('Total Energy (mAh)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(chart_dir, 'anomaly_analysis_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. 异常特征统计图
    plt.figure(figsize=(14, 10))
    
    # 统计各种异常特征
    anomaly_types = []
    for _, row in data.iterrows():
        features = analyze_process_anomaly(row)
        if features['high_foreground_duration']:
            anomaly_types.append('High Foreground Duration')
        if features['high_background_duration']:
            anomaly_types.append('High Background Duration')
        if features['high_foreground_energy']:
            anomaly_types.append('High Foreground Energy')
        if features['high_background_energy']:
            anomaly_types.append('High Background Energy')
        if features['high_foreground_energy_per_hour']:
            anomaly_types.append('High FG Energy/Hour')
        if features['high_background_energy_per_hour']:
            anomaly_types.append('High BG Energy/Hour')
        if features['high_total_energy_per_hour']:
            anomaly_types.append('High Total Energy/Hour')
        if features['low_battery']:
            anomaly_types.append('Low Battery')
    
    # 统计频次
    from collections import Counter
    anomaly_counts = Counter(anomaly_types)
    
    if anomaly_counts:
        plt.subplot(2, 1, 1)
        types = list(anomaly_counts.keys())
        counts = list(anomaly_counts.values())
        bars = plt.bar(range(len(types)), counts, color='coral', alpha=0.7, edgecolor='black')
        plt.title('Anomaly Feature Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Anomaly Type')
        plt.ylabel('Count')
        plt.xticks(range(len(types)), types, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom', fontweight='bold')
    
    # 6. 进程类型分析
    plt.subplot(2, 1, 2)
    process_types = []
    for name in data['name']:
        if 'game' in name.lower() or 'tmgp' in name.lower():
            process_types.append('Game')
        elif 'system' in name.lower() or 'android' in name.lower():
            process_types.append('System')
        elif 'com.' in name:
            process_types.append('App')
        else:
            process_types.append('Other')
    
    type_counts = Counter(process_types)
    if type_counts:
        types = list(type_counts.keys())
        counts = list(type_counts.values())
        colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
        bars = plt.bar(types, counts, color=colors[:len(types)], alpha=0.7, edgecolor='black')
        plt.title('Process Type Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Process Type')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(chart_dir, 'anomaly_features_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. 能耗效率分析热力图
    plt.figure(figsize=(12, 8))
    
    # 创建能耗效率矩阵
    energy_efficiency_data = data[['foreground_energy_per_hour', 'background_energy_per_hour', 
                                  'total_energy_per_hour', 'anomaly_score']].copy()
    
    # 计算相关性矩阵
    correlation_matrix = energy_efficiency_data.corr()
    
    # 绘制热力图
    sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0, 
                square=True, fmt='.3f', cbar_kws={'label': 'Correlation'})
    plt.title('Energy Efficiency Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(chart_dir, 'energy_efficiency_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"可视化图表已保存到: {chart_dir}")
    return chart_dir

def main():
    # 读取异常进程数据
    csv_path = '/mnt/ymj/vivo/异常进程/data/anomaly_processes.csv'
    
    try:
        data = pd.read_csv(csv_path, encoding='utf-8')
        print(f'异常进程总数为{len(data)}')
        print(f'本次分析的异常进程数量为{len(data)}')
    except FileNotFoundError:
        print(f"错误：找不到文件 {csv_path}")
        print("请先运行 anomaly_detection.py 生成数据")
        return
    
    # 确保结果目录存在
    result_dir = '/mnt/ymj/vivo/异常进程/result'
    os.makedirs(result_dir, exist_ok=True)
    
    # 初始化列表来存储分析结果
    json_list = []
    
    print("\n开始分析异常进程...")
    
    # 逐行处理数据
    for index, row in data.iterrows():
        # 分析进程异常特征
        anomaly_features = analyze_process_anomaly(row)
        
        # 生成异常描述
        anomaly_description = generate_anomaly_description(row, anomaly_features)
        
        # 生成分析报告
        analysis_report = generate_analysis_report(row, anomaly_features, anomaly_description)
        
        # 构建结果消息
        message = {
            "进程名称": row['name'],
            "设备型号": row['product'],
            "运行日期": row['event_date'],
            "前台时长_秒": row['foreground_duration'],
            "后台时长_秒": row['background_duration'],
            "前台能耗_mAh": row['foreground_energy'],
            "后台能耗_mAh": row['background_energy'],
            "前台单位时间能耗_mAh每小时": row['foreground_energy_per_hour'],
            "后台单位时间能耗_mAh每小时": row['background_energy_per_hour'],
            "总单位时间能耗_mAh每小时": row['total_energy_per_hour'],
            "设备电量": row['charge'],
            "异常分数": row['anomaly_score'],
            "异常特征": anomaly_description,
            "分析结果": analysis_report
        }
        json_list.append(message)
        
        # 打印进度
        if (index + 1) % 10 == 0 or index == len(data) - 1:
            print(f"已分析 {index + 1}/{len(data)} 个进程")
    
    # 保存结果
    output_df = pd.DataFrame(json_list)
    output_path = os.path.join(result_dir, '异常进程分析结果_simple.csv')
    output_df.to_csv(output_path, index=False, encoding='utf-8')
    
    # 保存JSON格式结果
    json_path = os.path.join(result_dir, '异常进程分析结果_simple.json')
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(json_list, json_file, ensure_ascii=False, indent=4)
    
    print(f'\n异常进程分析完成！')
    print(f'CSV文件已保存到: {output_path}')
    print(f'JSON文件已保存到: {json_path}')
    print(f'\n共分析了 {len(json_list)} 个异常进程')
    
    # 生成可视化图表
    print("\n正在生成可视化图表...")
    chart_dir = create_visualizations(data, result_dir)
    
    # 统计分析
    print('\n=== 异常类型统计 ===')
    anomaly_type_count = 0
    for message in json_list:
        if message['异常特征'] != '未检测到明显异常特征':
            anomaly_type_count += 1
            print(f"- {message['进程名称']}: {message['异常特征']}")
    
    print(f'\n检测到明显异常特征的进程: {anomaly_type_count}/{len(json_list)} ({anomaly_type_count/len(json_list)*100:.1f}%)')
    
    # 统计单位时间能耗异常情况
    print('\n=== 单位时间能耗异常统计 ===')
    high_energy_per_hour_count = 0
    for message in json_list:
        if ('单位时间能耗过高' in message['异常特征']):
            high_energy_per_hour_count += 1
            print(f"- {message['进程名称']}: 前台{message['前台单位时间能耗_mAh每小时']:.2f}mAh/h, 后台{message['后台单位时间能耗_mAh每小时']:.2f}mAh/h, 总计{message['总单位时间能耗_mAh每小时']:.2f}mAh/h")
    
    print(f'\n单位时间能耗异常进程数量: {high_energy_per_hour_count}/{len(json_list)} ({high_energy_per_hour_count/len(json_list)*100:.1f}%)')
    
    # 显示前5个最异常的进程
    print('\n=== 异常程度最高的5个进程 ===')
    sorted_data = data.sort_values('anomaly_score').head(5)
    for _, row in sorted_data.iterrows():
        features = analyze_process_anomaly(row)
        desc = generate_anomaly_description(row, features)
        print(f"- {row['name']}: 异常分数={row['anomaly_score']:.4f}, 特征={desc}")
    
    print(f"\n分析完成！结果文件和图表已保存到: {result_dir}")
    print(f"可视化图表目录: {chart_dir}")

if __name__ == "__main__":
    main()