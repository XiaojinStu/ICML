import os
import json
import glob
import argparse
from pathlib import Path

def extract_examples_from_task(task_dir, task_data, max_examples=None):
    """从task.json中提取examples，可限制最大数量"""
    examples = []
    name = task_data.get("name", "unknown")
    
    # 获取所有examples
    all_examples = task_data.get("examples", [])
    
    # 限制examples数量
    if max_examples is not None:
        all_examples = all_examples[:max_examples]
    
    for idx, example in enumerate(all_examples):
        example_id = f"{name}-{idx}"
        question = example.get("input", "")
        answer = example.get("target", "")
        
        # 如果target是数字，直接使用；如果是target_scores，找到值为1.0的键
        if not answer and "target_scores" in example:
            target_scores = example["target_scores"]
            for key, value in target_scores.items():
                if value == 1.0:
                    answer = key
                    break
        
        if question and answer:  # 只添加有效的例子
            examples.append({
                "id": example_id,
                "question": question,
                "answer": answer
            })
    
    return examples

def main(input_dir=None, output_file=None, max_per_file=None):
    """
    从文件夹中提取所有task.json文件中的examples
    
    Args:
        input_dir: 输入目录路径，默认为当前目录
        output_file: 输出JSON文件名，如果为None则自动生成 big_bench{数据条数}.json
        max_per_file: 每个文件最多提取的example数量，None表示全部提取
    """
    if input_dir is None:
        input_dir = os.getcwd()
    
    all_examples = []
    
    # 查找所有task.json文件
    task_files = glob.glob(os.path.join(input_dir, "**", "task.json"), recursive=True)
    
    if not task_files:
        print(f"在 {input_dir} 中未找到task.json文件")
        return
    
    print(f"找到 {len(task_files)} 个task.json文件")
    
    for task_file in task_files:
        try:
            with open(task_file, 'r', encoding='utf-8') as f:
                task_data = json.load(f)
            
            task_dir = os.path.dirname(task_file)
            folder_name = os.path.basename(task_dir)
            
            # 验证文件夹名是否与task.json中的name一致
            task_name = task_data.get("name", "")
            if task_name and folder_name != task_name:
                print(f"警告: 文件夹名 '{folder_name}' 与task.json中的name '{task_name}' 不一致")
            
            examples = extract_examples_from_task(task_dir, task_data, max_examples=max_per_file)
            all_examples.extend(examples)
            
            total_examples = len(task_data.get("examples", []))
            extracted_count = len(examples)
            
            if max_per_file and extracted_count < total_examples:
                print(f"从 {task_file} 中提取了 {extracted_count}/{total_examples} 个例子（限制每个文件最多 {max_per_file} 个）")
            else:
                print(f"从 {task_file} 中提取了 {extracted_count}/{total_examples} 个例子")
            
        except Exception as e:
            print(f"处理文件 {task_file} 时出错: {e}")
    
    # 自动生成输出文件名，格式为 big_bench{数据条数}.json
    if output_file is None:
        output_file = f"big_bench{len(all_examples)}.json"
    else:
        # 如果用户指定了文件名但文件名中不包含数据条数，我们可以添加数据条数信息
        # 这里我们按照用户指定的文件名保存，不自动添加数据条数
        pass
    
    # 保存到输出文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_examples, f, ensure_ascii=False, indent=2)
        
        print(f"\n成功提取 {len(all_examples)} 个例子到 {output_file}")
        if max_per_file:
            print(f"每个task.json文件最多提取 {max_per_file} 个例子")
        
        # 显示一些统计信息
        print("\n统计信息:")
        print(f"- 总数据条数: {len(all_examples)}")
        print(f"- 输出文件: {output_file}")
        print(f"- 文件大小: {os.path.getsize(output_file) / 1024:.2f} KB")
        
    except Exception as e:
        print(f"保存输出文件时出错: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从task.json文件中提取examples")
    parser.add_argument("-i", "--input-dir", help="输入目录路径（默认: 当前目录）")
    parser.add_argument("-o", "--output-file", default=None)
    parser.add_argument("-m", "--max-per-file", type=int, default=10)
    
    args = parser.parse_args()
    main(args.input_dir, args.output_file, args.max_per_file)