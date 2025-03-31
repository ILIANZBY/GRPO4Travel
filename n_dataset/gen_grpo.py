import csv
import json

def process_csv_to_json(input_csv, output_json):
    results = []
    
    with open(input_csv, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # 构建每个样本的prompt
            prompt_template = """您是一位熟练的规划师。
根据提供的信息和查询，请给我一个详细的计划，包括列车号（例如：T124）、住宿名称等具体信息。
请注意，您计划中的所有信息应来自提供的参考信息。
此外，所有细节应符合常识。符号 '-' 表示该信息不必要。例如，在提供的示例中，您不需要规划返回出发城市后的行程。当您在一天内前往两个城市时，应在"当前城市"部分注明，格式与示例中的相同（即，从 A 到 B）。

一些字段的输出格式解释
transportation: "TrainNumber: <TrainNumber>, <from_org_to_dest>, DepartureTime: <DepartureTime>, ArrivalTime: <ArrivalTime>"
attraction: "<Name>, <Address>"
accommodation: "<Name>"

您必须遵循示例中给出的格式。
请以 JSON 结构提供您的答案，如以下示例：

***** Example Starts *****
问题: 请设计一个从北京出发前往秦皇岛的旅行计划，为期3天，涵盖2024年4月1日至2024年4月3日，预算为2500人民币。
Travel Plan in JSON format:
{{
  'travel_plan': [
  {{'days': 1,
  'current_city': '从北京到秦皇岛',
  'transportation': 'TrainNumber: G9891, 从北京到秦皇岛, DepartureTime: 6:25, ArrivalTime: 08:56',
  'attraction': '鼓楼, 秦皇岛市山海关区龙海大道1号老龙头景区海神庙内',
  'breakfast': '任义烧烤(兴华商城1号楼店)',
  'lunch': '君临麻辣香锅(西顺城小区店)',
  'dinner': '',
  'accommodation': '山海关沐海安澜海景别墅(老龙头店)'}},
 {{'days': 2,
  'current_city': '秦皇岛',
  'transportation': '-',
  'attraction': '悬阳洞, 秦皇岛市山海关区三道关村附近长寿山景区内.',
  'breakfast': '小白楼汤馆(蓝天家园店)',
  'lunch': '老菜馆(教军场路店)',
  'dinner': '常品轩火锅烧烤',
  'accommodation': '山海关沐海安澜海景别墅(老龙头店)'}},
 {{'days': 3,
  'current_city': '从秦皇岛到北京',
  'transportation': 'TrainNumber: G9900, 从秦皇岛到北京, DepartureTime: 20:29, ArrivalTime: 22:10',
  'attraction': '孟姜女庙, 河北省秦皇岛市山海关区望夫石村.',
  'breakfast': '山海渔家(河南路)',
  'lunch': '依铭轩浑锅(唐子寨碧海龙源小区店)',
  'dinner': '金泽饭店',
  'accommodation': '-'}}
 ]
}}
***** Example Ends *****

给定可用的参考信息: {reference_info}
问题: {query}

请注意，您旅行计划中的所有信息（包括TrainNumber, accommodation, attraction, restaurants等）必须只能来自给定的参考信息中，同时您旅行计划中的餐厅，旅游景点不能出现重复。
请不要输出其他内容，请确保输出的内容可以被 json.loads() 解析。
Travel Plan in JSON format:"""
            
            # 替换模板中的变量
            prompt = prompt_template.format(
                reference_info=row['reference_information'],
                query=row['query']
            )
            
            # 添加到结果列表
            results.append({
                "prompt": prompt
            })
    
    # 写入JSON文件
    with open(output_json, 'w', encoding='utf-8') as jsonfile:
        json.dump(results, jsonfile, ensure_ascii=False, indent=2)

# 使用示例
input_csv = '/share/home/wuqingyao_zhangboyang/grpo/n_dataset/n_train.csv'  # 替换为你的CSV文件路径
output_json = '/share/home/wuqingyao_zhangboyang/grpo/n_dataset/n_grpo.json'  # 输出JSON文件路径
process_csv_to_json(input_csv, output_json)