import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子以确保结果可重现
np.random.seed(42)
random.seed(42)

def generate_random_data(n_samples=1000):
    """
    生成1000条随机的进程数据
    """
    data = []
    
    # 定义一些常见的进程名称
    process_names = [
        'com.android.chrome', 'com.tencent.mm', 'com.alibaba.android.rimet',
        'com.tencent.mobileqq', 'com.sina.weibo', 'com.taobao.taobao',
        'com.netease.cloudmusic', 'com.tencent.tmgp.sgame', 'com.eg.android.AlipayGphone',
        'com.ss.android.ugc.aweme', 'com.baidu.BaiduMap', 'com.autonavi.minimap',
        'com.jingdong.app.mall', 'com.sankuai.meituan', 'com.dianping.v1',
        'com.UCMobile', 'com.qiyi.video', 'com.youku.phone', 'com.tencent.qqlive',
        'com.android.settings', 'com.android.systemui', 'com.android.phone'
    ]
    
    # 定义设备型号
    products = ['vivo X90', 'vivo X80', 'vivo S15', 'vivo Y76s', 'vivo iQOO 9', 'vivo NEX 3S']
    
    # 定义系统版本
    sysversions = ['V13.1.2.0', 'V13.0.1.0', 'V12.5.3.0', 'V12.1.0.0', 'V11.2.1.0']
    
    for i in range(n_samples):
        # 生成基础设备信息
        pie_version = 'Android 9'
        product = random.choice(products)
        sysversion = random.choice(sysversions)
        vaid = f'VA{random.randint(100000, 999999)}'
        imei = f'{random.randint(100000000000000, 999999999999999)}'
        day = random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        
        # 生成时间信息
        start_time = datetime.now() - timedelta(days=random.randint(0, 30), 
                                               hours=random.randint(0, 23),
                                               minutes=random.randint(0, 59))
        
        # 生成持续时间（正常情况下前台时间较短，后台时间较长）
        if random.random() < 0.1:  # 10%的概率生成异常数据
            # 异常情况：前台时间过长或后台时间过短，能耗异常
            foreground_duration = random.randint(3600, 28800)  # 1-8小时前台
            background_duration = random.randint(0, 1800)      # 0-30分钟后台
            foreground_energy = random.randint(500, 2000)      # 异常高能耗
            background_energy = random.randint(200, 800)       # 异常高后台能耗
            charge = random.randint(10, 40)  # 电量消耗较大
        else:
            # 正常情况
            foreground_duration = random.randint(60, 1800)     # 1-30分钟前台
            background_duration = random.randint(1800, 14400)  # 30分钟-4小时后台
            foreground_energy = random.randint(10, 200)        # 正常能耗
            background_energy = random.randint(5, 100)         # 正常后台能耗
            charge = random.randint(60, 95)  # 正常电量
        
        end_time = start_time + timedelta(seconds=foreground_duration + background_duration)
        
        # 计算单位时间能耗（mAh/小时）
        foreground_energy_per_hour = (foreground_energy / (foreground_duration / 3600)) if foreground_duration > 0 else 0
        background_energy_per_hour = (background_energy / (background_duration / 3600)) if background_duration > 0 else 0
        total_energy = foreground_energy + background_energy
        total_duration_hours = (foreground_duration + background_duration) / 3600
        total_energy_per_hour = total_energy / total_duration_hours if total_duration_hours > 0 else 0
        
        data.append({
            'pie_version': pie_version,
            'product': product,
            'sysversion': sysversion,
            'vaid': vaid,
            'imei': imei,
            'day': day,
            'id': f'ID{i+1:06d}',
            'start_time': int(start_time.timestamp()),
            'end_time': int(end_time.timestamp()),
            'formatted_start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'formatted_end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
            'name': random.choice(process_names),
            'charge': charge,
            'screen': random.choice([0, 1]),  # 0=屏幕关闭, 1=屏幕开启
            'foreground_duration': foreground_duration,
            'background_duration': background_duration,
            'foreground_energy': foreground_energy,
            'background_energy': background_energy,
            'foreground_energy_per_hour': round(foreground_energy_per_hour, 2),
            'background_energy_per_hour': round(background_energy_per_hour, 2),
            'total_energy_per_hour': round(total_energy_per_hour, 2),
            'event_date': start_time.strftime('%Y-%m-%d')
        })
    
    return pd.DataFrame(data)

def detect_anomalies(df):
    """
    使用孤立森林算法检测异常进程
    """
    # 选择用于异常检测的特征，包含单位时间能耗
    features = [
        'foreground_duration', 'background_duration', 
        'foreground_energy', 'background_energy', 'charge',
        'foreground_energy_per_hour', 'background_energy_per_hour', 'total_energy_per_hour'
    ]
    
    # 提取特征数据
    X = df[features].copy()
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 使用孤立森林进行异常检测
    # contamination参数设置为0.1，表示预期10%的数据为异常
    iso_forest = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
    anomaly_labels = iso_forest.fit_predict(X_scaled)
    
    # 获取异常分数
    anomaly_scores = iso_forest.decision_function(X_scaled)
    
    # 添加结果到数据框
    df['anomaly_label'] = anomaly_labels  # -1表示异常，1表示正常
    df['anomaly_score'] = anomaly_scores  # 分数越低越异常
    df['is_anomaly'] = df['anomaly_label'] == -1
    
    return df, iso_forest, scaler

def analyze_anomalies(df):
    """
    分析异常数据的特征
    """
    anomalies = df[df['is_anomaly'] == True]
    normal = df[df['is_anomaly'] == False]
    
    print(f"总数据量: {len(df)}")
    print(f"异常数据量: {len(anomalies)} ({len(anomalies)/len(df)*100:.2f}%)")
    print(f"正常数据量: {len(normal)} ({len(normal)/len(df)*100:.2f}%)")
    print("\n异常数据统计:")
    
    features = [
        'foreground_duration', 'background_duration', 'foreground_energy', 'background_energy', 'charge',
        'foreground_energy_per_hour', 'background_energy_per_hour', 'total_energy_per_hour'
    ]
    
    feature_names = {
        'foreground_duration': '前台运行时长(秒)',
        'background_duration': '后台运行时长(秒)',
        'foreground_energy': '前台能耗(mAh)',
        'background_energy': '后台能耗(mAh)',
        'charge': '设备电量(%)',
        'foreground_energy_per_hour': '前台单位时间能耗(mAh/小时)',
        'background_energy_per_hour': '后台单位时间能耗(mAh/小时)',
        'total_energy_per_hour': '总单位时间能耗(mAh/小时)'
    }
    
    for feature in features:
        print(f"\n{feature_names[feature]}:")
        print(f"  异常数据 - 均值: {anomalies[feature].mean():.2f}, 标准差: {anomalies[feature].std():.2f}")
        print(f"  正常数据 - 均值: {normal[feature].mean():.2f}, 标准差: {normal[feature].std():.2f}")
    
    return anomalies, normal

def save_results(df, anomalies):
    """
    保存结果到文件
    """
    # 保存完整数据到data目录
    df.to_csv('/mnt/ymj/vivo/异常进程/data/all_process_data.csv', index=False, encoding='utf-8')
    
    # 保存异常数据到data目录
    anomalies.to_csv('/mnt/ymj/vivo/异常进程/data/anomaly_processes.csv', index=False, encoding='utf-8')
    
    print(f"\n完整数据已保存到: /mnt/ymj/vivo/异常进程/data/all_process_data.csv")
    print(f"异常数据已保存到: /mnt/ymj/vivo/异常进程/data/anomaly_processes.csv")

def main():
    print("开始生成随机数据...")
    df = generate_random_data(1000)
    
    print("\n开始异常检测...")
    df, model, scaler = detect_anomalies(df)
    
    print("\n分析异常数据...")
    anomalies, normal = analyze_anomalies(df)
    
    print("\n保存结果...")
    save_results(df, anomalies)
    
    print("\n异常检测完成！")
    
    # 显示一些异常样本
    print("\n异常进程样本:")
    print(anomalies[['name', 'foreground_duration', 'background_duration', 
                    'foreground_energy', 'background_energy', 'anomaly_score']].head(10))

if __name__ == "__main__":
    main()