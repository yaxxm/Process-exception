import json
import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import ast
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 强制将torch设置为使用第0号GPU
torch.cuda.set_device(0)

# 读取异常进程数据
csv_path = '/mnt/ymj/vivo/异常进程/data/anomaly_processes.csv'  # 异常进程数据路径
data = pd.read_csv(csv_path, encoding='utf-8')
print(f'异常进程总数为{len(data)}')
print(f'本次分析的异常进程数量为{len(data)}')

def predict(messages, model, tokenizer):
    device = "cuda"

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt", padding=True).to(device)  # 添加 padding=True

    with torch.no_grad():  # 禁用梯度计算
        generated_ids = model.generate(model_inputs.input_ids, 
                                       attention_mask=model_inputs.attention_mask,  # 显式传递 attention_mask
                                       max_new_tokens=512)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


def clean_and_merge_response(response):
    # 保留换行符，只替换多余的回车符
    response = response.replace('\r', '')
    return response

def analyze_process_anomaly(row):
    """
    分析单个进程的异常特征
    """
    analysis = {
        'high_foreground_duration': row['foreground_duration'] > 3600,  # 前台时间超过1小时
        'high_background_duration': row['background_duration'] > 14400,  # 后台时间超过4小时
        'high_foreground_energy': row['foreground_energy'] > 500,  # 前台能耗过高
        'high_background_energy': row['background_energy'] > 200,  # 后台能耗过高
        'high_foreground_energy_per_hour': row['foreground_energy_per_hour'] > 300,  # 前台单位时间能耗过高
        'high_background_energy_per_hour': row['background_energy_per_hour'] > 100,  # 后台单位时间能耗过高
        'high_total_energy_per_hour': row['total_energy_per_hour'] > 200,  # 总单位时间能耗过高
        'low_battery': row['charge'] < 50,  # 电量过低
        'anomaly_severity': abs(row['anomaly_score'])  # 异常严重程度
    }
    return analysis



model_dir = "/mnt/ymj/GLM-4-main/THUDM/glm-4-9b-chat"
# lora_dir = "/mnt/data/six/ymj/biaozhu/task6/checkpoint-600"

# 加载原下载路径的tokenizer和model
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cuda:0", torch_dtype=torch.bfloat16, trust_remote_code=True)

# 加载训练好的Lora模型
# model = PeftModel.from_pretrained(model, model_id=lora_dir)

# 将模型设置为评估模式
model.eval()

# 初始化列表来存储JSON数据
json_list = []

# 逐行处理数据并将结果添加到列表中


before_row = None
for index, row in tqdm(data.iterrows(), total=len(data)):
    #------------------推理---------------------------
    input_text = row['name']  # 使用进程名作为唯一标识符
    
    # 分析进程异常特征
    anomaly_features = analyze_process_anomaly(row)
    
    # 构建异常特征描述
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
    
    anomaly_description = "、".join(anomaly_desc) if anomaly_desc else "未检测到明显异常特征"
    
    test_texts = {
        "instruction": "你是一个专业的Android系统性能分析师，请根据以下进程的运行数据，分析该进程为什么被标记为异常。请按照以下格式进行分析：\n\n1. 进程基本信息分析\n2. 运行时间异常分析\n3. 能耗异常分析\n4. 设备状态分析\n5. 异常原因总结\n6. 优化建议",
        "input": f"异常进程数据：\n\n进程名称：{row['name']}\n设备型号：{row['product']}\n系统版本：{row['sysversion']}\n运行日期：{row['event_date']}\n运行时段：{row['formatted_start_time']} 至 {row['formatted_end_time']}\n\n运行时间数据：\n- 前台运行时长：{row['foreground_duration']}秒 ({row['foreground_duration']/60:.1f}分钟)\n- 后台运行时长：{row['background_duration']}秒 ({row['background_duration']/60:.1f}分钟)\n- 屏幕状态：{'开启' if row['screen'] == 1 else '关闭'}\n\n能耗数据：\n- 前台能耗：{row['foreground_energy']}mAh\n- 后台能耗：{row['background_energy']}mAh\n- 前台单位时间能耗：{row['foreground_energy_per_hour']:.2f}mAh/小时\n- 后台单位时间能耗：{row['background_energy_per_hour']:.2f}mAh/小时\n- 总单位时间能耗：{row['total_energy_per_hour']:.2f}mAh/小时\n- 设备电量：{row['charge']}%\n\n异常检测结果：\n- 异常分数：{row['anomaly_score']:.4f} (分数越低越异常)\n- 检测到的异常特征：{anomaly_description}\n\n请分析该进程异常的具体原因并提供优化建议。"
    }
    instruction = test_texts['instruction']
    input_value = test_texts['input']

    if before_row is None or input_text != before_row:
        messages_1 = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_value}
        ]

        response = predict(messages_1, model, tokenizer)  # 推理结果喂response
        response = response.strip()
        print(f"\n分析进程: {row['name']}")
        print(response)
        response = clean_and_merge_response(response)  # 清理并合并可能的多重列表

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
        "分析结果": response
    }
    json_list.append(message)
    before_row = input_text  # 暂存上一组的文本

# 将数据转换为DataFrame并保存为CSV文件到result目录
output_df = pd.DataFrame(json_list)
output_path = '/mnt/ymj/vivo/异常进程/result/LLM/异常进程分析结果.csv'
output_df.to_csv(output_path, index=False, encoding='utf-8')

# 同时保存一份JSON格式的结果到result目录
json_path = '/mnt/ymj/vivo/异常进程/result/LLM/异常进程分析结果.json'
with open(json_path, 'w', encoding='utf-8') as json_file:
    json.dump(json_list, json_file, ensure_ascii=False, indent=4)

print(f'\n异常进程分析完成！')
print(f'CSV文件已保存到: {output_path}')
print(f'JSON文件已保存到: {json_path}')
print(f'\n共分析了 {len(json_list)} 个异常进程')

# 统计异常类型
print('\n异常类型统计:')
for message in json_list:
    if message['异常特征'] != '未检测到明显异常特征':
        print(f"- {message['进程名称']}: {message['异常特征']}")

# 统计单位时间能耗异常情况
print('\n单位时间能耗异常统计:')
high_energy_per_hour_count = 0
for message in json_list:
    if ('单位时间能耗过高' in message['异常特征']):
        high_energy_per_hour_count += 1
        print(f"- {message['进程名称']}: 前台{message['前台单位时间能耗_mAh每小时']:.2f}mAh/h, 后台{message['后台单位时间能耗_mAh每小时']:.2f}mAh/h, 总计{message['总单位时间能耗_mAh每小时']:.2f}mAh/h")

print(f'\n单位时间能耗异常进程数量: {high_energy_per_hour_count}/{len(json_list)} ({high_energy_per_hour_count/len(json_list)*100:.1f}%)')
