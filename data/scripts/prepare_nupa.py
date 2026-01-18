import json

def process_nubp_data(input_file, output_file):
    # 读取原始JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 初始化结果列表和计数器
    result = []
    index = 0
    
    # 遍历原始数据的结构
    for category, subcategories in data.items():
        for subcategory, questions in subcategories.items():
            for question in questions:
                # 分离问题和答案（以等号为分隔符）
                if '=' in question:
                    # 分割字符串，最后一部分是答案
                    parts = question.rsplit('=', 1)
                    question_text = parts[0].strip()
                    answer = parts[1].strip()
                    
                    # 创建新的条目
                    entry = {
                        "id": f"nubp-test-{index}",
                        "question": question_text,
                        "answer": answer
                    }
                    
                    result.append(entry)
                    index += 1
    
    # 写入新的JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"处理完成！共转换了 {index} 条数据")
    print(f"结果已保存到 {output_file}")

# 使用示例
if __name__ == "__main__":
    input_file = "./nupa_test_ori.json"
    output_file = "./nupa_test_all.json"
    process_nubp_data(input_file, output_file)