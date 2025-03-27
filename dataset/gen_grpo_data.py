import json

# 读取输入的JSON文件
def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# 处理数据并提取需要的字段
def process_data(data):
    result = []
    for item in data:
        prompt = item.get("instruction", "")
        reference = json.dumps(item.get("output", {}), ensure_ascii=False)
        result.append({
            "prompt": prompt,
            "reference": reference
        })
    return result

# 将处理后的数据写入新的JSON文件
def write_json_file(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# 主函数
def main():
    input_file_path = "/share/home/wuqingyao_zhangboyang/grpo/dataset/3000_sft.json"  # 输入JSON文件路径
    output_file_path = "/share/home/wuqingyao_zhangboyang/grpo/dataset/3000_grpo.json"  # 输出JSON文件路径

    # 读取输入文件
    input_data = read_json_file(input_file_path)

    # 处理数据
    processed_data = process_data(input_data)

    # 写入输出文件
    write_json_file(output_file_path, processed_data)

    print(f"处理完成，结果已保存到 {output_file_path}")

if __name__ == "__main__":
    main()