import json
import os
import glob

def process_json_file(file_path, global_counter):
    """处理单个JSON文件并转换为目标格式"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []
    
    for item in data.get('data', []):
        # 生成全局唯一的ID
        item_id = f"arithmetic_operation-{global_counter[0]}"
        global_counter[0] += 1
        
        # 解析struct_data并构建question
        struct_data_str = item.get('struct_data', '{}')
        try:
            # 使用更安全的方式解析struct_data
            import ast
            struct_data = ast.literal_eval(struct_data_str.replace("'", '"'))
            # 将struct_data转换为字符串格式
            struct_items = []
            for key, value in struct_data.items():
                struct_items.append(f"{key}: {value}")
            struct_text = ", ".join(struct_items)
        except:
            struct_text = struct_data_str
        
        # 构建question
        question_text = f"{item.get('question', '')} {struct_text}"
        
        # 获取answer
        answer = item.get('answer', '')
        
        # 构建结果字典
        result = {
            "id": item_id,
            "question": question_text.strip(),
            "answer": answer
        }
        
        results.append(result)
    
    return results

def process_all_json_files(folder_path):
    """处理文件夹下所有JSON文件"""
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    
    all_results = []
    global_counter = [0]  # 使用列表以便在函数中修改
    
    for json_file in json_files:
        print(f"正在处理文件: {json_file}")
        results = process_json_file(json_file, global_counter)
        all_results.extend(results)
    
    return all_results

def save_results(results, output_file):
    """保存结果到JSON文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"转换完成！共处理 {len(results)} 条数据")
    print(f"结果已保存到: {output_file}")

# 使用示例
if __name__ == "__main__":
    # 指定文件夹路径
    folder_path = "../NumericBench/arithmetic_operation"  # 当前文件夹，可以修改为您的文件夹路径
    
    # 处理所有JSON文件
    results = process_all_json_files(folder_path)
    
    # 保存结果
    if results:
        save_results(results, "NumericBench_all.json")
        
        # 打印前几条结果作为示例
        print("\n前10条转换结果:")
        for i, result in enumerate(results[:10]):
            print(f"{i+1}. ID: {result['id']}")
            print(f"   问题: {result['question']}")
            print(f"   答案: {result['answer']}")
            print()
    else:
        print("未找到JSON文件或文件格式不正确")