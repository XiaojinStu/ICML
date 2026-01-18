import json

def process_json_file(input_file, output_file, id_prefix="math401"):
    """
    处理JSON文件，将每行转换为新的格式
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        id_prefix: ID前缀，默认为"math401"
    """
    result = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
                
            try:
                # 解析每行的JSON
                data = json.loads(line)
                

                new_item = {
                    "id": f"{id_prefix}-{i}",
                    "question": data.get("query", ""),
                    "answer": data.get("response", "")
                }
                
                result.append(new_item)
                
            except json.JSONDecodeError as e:
                print(f"第{i+1}行JSON解析错误: {e}")
                continue
    

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成！共处理 {len(result)} 条数据")
    print(f"结果已保存到: {output_file}")
    
    return result


if __name__ == "__main__":

    input_file = "../math401-llm/math401.json"  
    output_file = "../math401_all.json" 
    
    result = process_json_file(input_file, output_file)
    
    for item in result[:3]:
        print(f"id: {item['id']}")
        print(f"question: {item['question']}")
        print(f"answer: {item['answer']}")
        print("-" * 30)