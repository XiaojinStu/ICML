import json
import random

def sample_nubp_data(input_file, output_file, sample_size=100, seed=42):
    """
    从每个类别中抽样指定数量的问题
    
    Args:
        input_file: 输入JSON文件路径
        output_file: 输出JSON文件路径
        sample_size: 每个类别抽样的数量，默认为100
        seed: 随机种子，确保结果可重复
    """
    # 设置随机种子
    random.seed(seed)
    
    # 读取原始JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 初始化结果列表和计数器
    result = []
    global_index = 0
    
    # 遍历每个类别
    for category, subcategories in data.items():
        print(f"处理类别: {category}")
        
        # 收集该类别下的所有问题
        all_questions_in_category = []
        
        # 遍历子类别
        for subcategory, questions in subcategories.items():
            # 处理每个问题
            for question_str in questions:
                if '=' in question_str:
                    # 分割问题和答案
                    parts = question_str.rsplit('=', 1)
                    question_text = parts[0].strip()
                    answer = parts[1].strip()
                    
                    # 添加到该类别的问题列表中
                    all_questions_in_category.append({
                        "question": question_text,
                        "answer": answer,
                        "category": category,
                        "subcategory": subcategory
                    })
        
        # 从该类别中抽样
        if len(all_questions_in_category) <= sample_size:
            # 如果问题数量少于等于样本大小，全部使用
            sampled_questions = all_questions_in_category
            print(f"  类别 '{category}' 共有 {len(all_questions_in_category)} 个问题，全部使用")
        else:
            # 否则随机抽样指定数量
            sampled_questions = random.sample(all_questions_in_category, sample_size)
            print(f"  类别 '{category}' 共有 {len(all_questions_in_category)} 个问题，随机抽样 {sample_size} 个")
        
        # 将抽样的问题添加到结果中
        for item in sampled_questions:
            result.append({
                "id": f"nubp-sampled-{global_index}",
                "question": item["question"],
                "answer": item["answer"],
                "category": item["category"],
                "subcategory": item["subcategory"]
            })
            global_index += 1
    
    # 写入新的JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n抽样完成！")
    print(f"共处理了 {len(data)} 个类别")
    print(f"总共抽样了 {len(result)} 个问题")
    print(f"结果已保存到 {output_file}")
    
    return result

def sample_and_summarize(input_file, output_file, sample_size=100):
    """
    抽样并生成统计信息
    """
    result = sample_nubp_data(input_file, output_file, sample_size)
    
    # 生成统计信息
    category_stats = {}
    for item in result:
        category = item["category"]
        if category not in category_stats:
            category_stats[category] = 0
        category_stats[category] += 1
    
    print("\n统计信息:")
    for category, count in category_stats.items():
        print(f"  {category}: {count} 个问题")
    
    # 保存统计信息到文件
    stats_file = output_file.replace(".json", "_stats.txt")
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("抽样统计信息\n")
        f.write("=" * 50 + "\n")
        f.write(f"总问题数: {len(result)}\n")
        f.write(f"类别数: {len(category_stats)}\n")
        f.write("=" * 50 + "\n")
        for category, count in category_stats.items():
            f.write(f"{category}: {count} 个问题\n")
    
    print(f"\n统计信息已保存到 {stats_file}")
    
    return result, category_stats

def sample_without_category_info(input_file, output_file, sample_size=100, seed=42):
    """
    抽样但不包含类别信息（与原始要求的格式完全一致）
    """
    # 设置随机种子
    random.seed(seed)
    
    # 读取原始JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 初始化结果列表和计数器
    result = []
    global_index = 0
    
    # 遍历每个类别
    for category, subcategories in data.items():
        # 收集该类别下的所有问题
        all_questions_in_category = []
        
        # 遍历子类别
        for subcategory, questions in subcategories.items():
            # 处理每个问题
            for question_str in questions:
                if '=' in question_str:
                    # 分割问题和答案
                    parts = question_str.rsplit('=', 1)
                    question_text = parts[0].strip()
                    answer = parts[1].strip()
                    
                    # 添加到该类别的问题列表中
                    all_questions_in_category.append((question_text, answer))
        
        # 从该类别中抽样
        if len(all_questions_in_category) <= sample_size:
            sampled_questions = all_questions_in_category
        else:
            sampled_questions = random.sample(all_questions_in_category, sample_size)
        
        # 将抽样的问题添加到结果中
        for question_text, answer in sampled_questions:
            result.append({
                "id": f"nubp-sampled-{global_index}",
                "question": question_text,
                "answer": answer
            })
            global_index += 1
    
    # 写入新的JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"抽样完成！总共抽样了 {len(result)} 个问题")
    print(f"结果已保存到 {output_file}")
    
    return result

# 主程序
if __name__ == "__main__":
    input_file = "nupa_test_ori.json"
    
    # 选项1：抽样并包含类别信息（便于分析）
    output_file1 = "nupa_test_440.json"
    print("选项1：抽样并包含类别信息")
    print("=" * 50)
    result1, stats1 = sample_and_summarize(input_file, output_file1, sample_size=10)
    
    # # 选项2：抽样但不包含类别信息（与原始格式完全一致）
    # output_file2 = "nubp_test_sampled.json"
    # print("\n\n选项2：抽样但不包含类别信息")
    # print("=" * 50)
    # result2 = sample_without_category_info(input_file, output_file2, sample_size=100)
    
    # 显示抽样结果的示例
    print("\n\n抽样结果示例（前5条）：")
    for i in range(min(5, len(result1))):
        print(f"\nID: {result1[i]['id']}")
        print(f"问题: {result1[i]['question']}")
        print(f"答案: {result1[i]['answer']}")