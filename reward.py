import json
import math
import pandas as pd
from typing import Dict, Any,List

CSV_FILE_PATH = "/share/home/wuqingyao_zhangboyang/grpo/dataset/3000.csv"
def load_csv_data() -> pd.DataFrame:
    """加载CSV文件数据"""
    return pd.read_csv(CSV_FILE_PATH)

def create_test_entry(row: pd.Series) -> Dict[str, Any]:
    """从CSV行创建测试条目"""
    return {
        "org": row["org"],
        "dest": row["dest"],
        "days": row["days"],
        "visiting_city_number": row["visiting_city_number"],
        "date": row["date"],
        "people_number": row["people_number"],
        "local_constraint": row["local_constraint"],
        "budget": row["budget"],
        "query": row["query"],
        "level": row["level"],
        "reference_information": row["reference_information"]
    }

def reward_function( prompts,completions:  List[str],**kwargs) -> Dict[int, float]:
    """
    计算模型输出的奖励分数，基于CSV文件中的样本数据和模型输出
    
    参数:
        csv_file_path: 包含样本数据的CSV文件路径
        model_outputs: 字典，键为样本索引，值为模型输出字符串
        
    返回:
        Dict[int, float]: 字典，键为样本索引，值为对应的奖励分数(0-1)
    """
    # 加载CSV数据
    df = load_csv_data()
    rewards = {}
    
    for idx, model_output in enumerate(completions):
        if idx >= len(df):
            continue  # 跳过无效索引
            
        row = df.iloc[idx]
        test_entry = create_test_entry(row)
        
        try:
            # 解析模型输出
            predicted_entry = json.loads(model_output)
        except json.JSONDecodeError:
            rewards[idx] = 0.0  # 无效的JSON格式
            continue
            
        # 初始化奖励分数
        total_score = 0.0
        max_possible_score = 0.0
        
        # 1. 检查天数是否有效 (权重: 0.1)
        max_possible_score += 0.1
        if is_valid_days(test_entry, {"plan": predicted_entry["travel_plan"]}):
            total_score += 0.1
        
        # 2. 检查城市访问顺序是否合理 (权重: 0.1)
        max_possible_score += 0.1
        if is_reasonalbe_visiting_city(test_entry, {"plan": predicted_entry["travel_plan"]}):
            total_score += 0.1
        
        # 3. 检查景点是否有效 (权重: 0.15)
        max_possible_score += 0.15
        if is_valid_attractions(test_entry, {"plan": predicted_entry["travel_plan"]}):
            total_score += 0.15
        
        # 4. 检查交通安排是否有效 (权重: 0.2)
        max_possible_score += 0.2
        if is_valid_transportation(test_entry, {"plan": predicted_entry["travel_plan"]}):
            total_score += 0.2
        
        # 5. 检查餐厅是否有效且不重复 (权重: 0.15)
        max_possible_score += 0.15
        if is_valid_restaurants(test_entry, {"plan": predicted_entry["travel_plan"]}):
            total_score += 0.15
        
        # 6. 检查住宿是否有效 (权重: 0.1)
        max_possible_score += 0.1
        if is_valid_accommodation(test_entry, {"plan": predicted_entry["travel_plan"]}):
            total_score += 0.1
        
        # 7. 检查预算是否合理 (权重: 0.2)
        max_possible_score += 0.2
        if is_valid_budget(test_entry, {"plan": predicted_entry["travel_plan"]}):
            total_score += 0.2
        
        # 计算最终奖励分数 (归一化到0-1)
        final_score = total_score / max_possible_score if max_possible_score > 0 else 0
        rewards[idx] = final_score
    
    return rewards

# 以下是评估脚本中的辅助函数，保持不变
def is_valid_days(test_entry: Dict[str, Any], predicted_entry: Dict[str, Any]) -> bool:
    if test_entry["days"] != len(predicted_entry["plan"]):
        return False
    return True

# def is_reasonalbe_visiting_city(test_entry: Dict[str, Any], predicted_entry: Dict[str, Any]) -> bool:
#     plan = predicted_entry["plan"]
#     citys = [item["current_city"] for item in plan]
    
#     org_city = test_entry["org"]
#     dest_city = test_entry["dest"]
    
#     for i, city in enumerate(citys):
#         if i == 0:
#             if city != f"从{org_city}到{dest_city}" and org_city not in city.split("到")[0]:
#                 return False
#         elif i == len(citys) - 1:
#             if city != f"从{dest_city}到{org_city}" and dest_city not in city.split("到")[0]:
#                 return False
#         else:
#             if city != dest_city:
#                 return False
#     return True

def is_reasonalbe_visiting_city(test_entry: Dict[str, Any], predicted_entry: Dict[str, Any]) -> bool:
    plan = predicted_entry["plan"]
    cities = [item["current_city"] for item in plan]
    
    org, dest = test_entry["org"], test_entry["dest"]
    expected_start = f"从{org}到{dest}"
    expected_end = f"从{dest}到{org}"
    
    # 检查首尾两天
    if cities[0] != expected_start or cities[-1] != expected_end:
        return False
    
    # 检查中间天数是否全部为目的地
    for city in cities[1:-1]:
        if city != dest:
            return False
    
    return True

# def is_valid_attractions(test_entry: Dict[str, Any], predicted_entry: Dict[str, Any]) -> bool:
#     plan = predicted_entry["plan"]
#     planned_attractions = [item["attraction"].split(",")[0] 
#                           for item in plan if item["attraction"] != "-"]
    
#     dest_city = test_entry["dest"]
#     attractions = json.loads(test_entry["reference_information"])[f"在{dest_city}的景点"]
#     attractions = [attr["Name"] for attr in attractions]
    
#     for attr in planned_attractions:
#         if attr not in attractions:
#             return False
#     return True

def is_valid_attractions(test_entry: Dict[str, Any], predicted_entry: Dict[str, Any]) -> bool:
    plan = predicted_entry["plan"]
    dest = test_entry["dest"]
    reference_attractions = json.loads(test_entry["reference_information"])[f"在{dest}的景点"]
    valid_attractions = {attr["Name"].split('(')[0].strip() for attr in reference_attractions}
    
    for day in plan:
        # 确保attraction是字符串类型
        attractions_str = day["attraction"] if isinstance(day["attraction"], str) else ", ".join(day["attraction"])
        attractions = [attr.split(',')[0].split('(')[0].strip() 
                      for attr in attractions_str.split(', ') if attr != "-"]
        for attr in attractions:
            if attr not in valid_attractions:
                return False
    return True

def is_valid_restaurants(test_entry: Dict[str, Any], predicted_entry: Dict[str, Any]) -> bool:
    restaurants_list = []
    plan = predicted_entry["plan"]
    
    for i, unit in enumerate(plan):
        # 处理lunch
        lunch = unit.get('lunch', '-')
        if isinstance(lunch, list):
            lunch = lunch[0] if lunch else '-'
        if lunch and lunch != '-':
            if lunch not in restaurants_list:
                restaurants_list.append(lunch)
            else:
                return False
        
        # 处理dinner
        dinner = unit.get('dinner', '-')
        if isinstance(dinner, list):
            dinner = dinner[0] if dinner else '-'
        if dinner and dinner != '-':
            if dinner not in restaurants_list:
                restaurants_list.append(dinner)
            else:
                return False
    
    return True

# def parse_transport_info(info_string: str) -> Dict[str, Any]:
#     parts = [part.strip() for part in info_string.split(',')]
    
#     transport_info = {}
    
#     for part in parts:
#         if ':' in part:
#             key, value = part.split(':', 1)
#             key = key.strip()
#             value = value.strip()
            
#             if key == "TrainNumber":
#                 value = value.split(',')[0].strip()
            
#             transport_info[key] = value
            
#     return transport_info

def parse_transport_info(info_string: str) -> Dict[str, Any]:
    parts = [part.strip() for part in info_string.split(',')]
    transport_info = {}
    route = []  # 新增：提取路线信息
    
    for part in parts:
        if ':' in part:
            key, value = part.split(':', 1)
            key, value = key.strip(), value.strip()
            if key == "TrainNumber":
                value = value.split(',')[0].strip()
            transport_info[key] = value
        elif "到" in part:  # 捕获路线信息（如"从青岛到秦皇岛"）
            route.append(part)
    
    if route:
        transport_info["Route"] = route[0]  # 记录路线信息
    
    return transport_info

def is_valid_transportation(test_entry: Dict[str, Any], predicted_entry: Dict[str, Any]) -> bool:
    plan = predicted_entry["plan"]
    planned_transportation = [parse_transport_info(item["transportation"]) 
                             if item["transportation"].strip() != "-" else "-"
                             for item in plan]
    
    org_city = test_entry["org"]
    dest_city = test_entry["dest"]

    # 获取火车的参考信息
    go_transportations = json.loads(test_entry["reference_information"])[f"从{org_city}到{dest_city}的列车"]
    go_train_num = [train["TrainNumber"] for train in go_transportations]
    go_departure_time = [train["DepartureTime"] for train in go_transportations]
    go_arrival_time = [train["ArrivalTime"] for train in go_transportations]
    
    return_transportations = json.loads(test_entry["reference_information"])[f"从{dest_city}到{org_city}的列车"]
    return_train_num = [train["TrainNumber"] for train in return_transportations]
    return_departure_time = [train["DepartureTime"] for train in return_transportations]
    return_arrival_time = [train["ArrivalTime"] for train in return_transportations]

    # 获取航班的参考信息
    go_flight = json.loads(test_entry["reference_information"])[f"从{org_city}到{dest_city}的航班"]
    go_flights_number = [flight["FlightNumber"] for flight in go_flight]
    go_flight_departure_time = [flight["DepTime"] for flight in go_flight]
    go_flight_arrival_time = [flight["ArrTime"] for flight in go_flight]

    return_flight = json.loads(test_entry["reference_information"])[f"从{dest_city}到{org_city}的航班"]
    return_flights_number = [flight["FlightNumber"] for flight in return_flight]
    return_flight_departure_time = [flight["DepTime"] for flight in return_flight]
    return_flight_arrival_time = [flight["ArrTime"] for flight in return_flight]
    
    try:
        for i, trans in enumerate(planned_transportation):
            if "TrainNumber" in trans:
                if i == 0:
                    if trans == "-":
                        return False
                    if (trans["TrainNumber"] not in go_train_num
                        or trans["DepartureTime"] not in go_departure_time
                        or trans["ArrivalTime"] not in go_arrival_time):
                        return False
                elif i == len(planned_transportation) - 1:
                    if trans == "-":
                        return False
                    if (trans["TrainNumber"] not in return_train_num
                        or trans["DepartureTime"] not in return_departure_time
                        or trans["ArrivalTime"] not in return_arrival_time):
                        return False
                else:
                    if trans != "-":
                        return False
                    
            elif "FlightNumber" in trans:
                if i == 0:
                    if trans == "-":
                        return False
                    if(trans["FlightNumber"] not in go_flights_number
                       or trans["DepTime"] not in go_flight_departure_time
                       or trans["ArrTime"] not in go_flight_arrival_time):
                        return False
                elif i == len(planned_transportation) - 1:
                    if trans == "-":
                        return False
                    if (trans["FlightNumber"] not in return_flights_number
                       or trans["DepTime"] not in return_flight_departure_time
                       or trans["ArrTime"] not in return_flight_arrival_time):
                        return False
                else:
                    if trans != "-":
                        return False
    
    except Exception as e:
        return False
    return True

# def parse_accommodation_info(info_string: str) -> Dict[str, Any]:
#     if info_string.strip() == "-":
#         return "-"
    
#     parts = [part.strip() for part in info_string.split(',')]
#     if len(parts) == 1:
#         return {"HotelName": parts[0], "HouseRules": None}
#     return {"HotelName": parts[0], "HouseRules": parts[1]}

def parse_accommodation_info(info_string: str) -> str:
    """直接返回酒店名称字符串，忽略房规信息"""
    if info_string.strip() == "-":
        return "-"
    # 提取酒店名称（假设名称在逗号或括号前）
    name = info_string.split(',')[0].split('(')[0].strip()
    return name



def is_valid_accommodation(test_entry: Dict[str, Any], predicted_entry: Dict[str, Any]) -> bool:
    plan = predicted_entry["plan"]
    accommodations = json.loads(test_entry["reference_information"])[f"在{test_entry['dest']}的酒店"]
    valid_hotels = {acc["HotelName"].split('(')[0].strip() for acc in accommodations}  # 统一处理名称格式
    
    for i, day_plan in enumerate(plan):
        acc_name = parse_accommodation_info(day_plan["accommodation"])
        # 最后一天允许无住宿，其他天必须有效
        if i != len(plan)-1 and (acc_name == "-" or acc_name not in valid_hotels):
            return False
    return True

# def get_total_cost(test_entry: Dict[str, Any], predicted_entry: Dict[str, Any]) -> float:
#     plan = predicted_entry["plan"]
    
#     # 计算交通费用
#     planned_transportation = [parse_transport_info(item["transportation"]) 
#                              for item in plan if item["transportation"].strip() != "-"]
    
#     org_city = test_entry["org"]
#     dest_city = test_entry["dest"]
    
#     go_transportations = json.loads(test_entry["reference_information"])[f"从{org_city}到{dest_city}的列车"]
#     go_price_map = {train["TrainNumber"]: train["Price"] for train in go_transportations}
    
#     return_transportations = json.loads(test_entry["reference_information"])[f"从{dest_city}到{org_city}的列车"]
#     return_price_map = {train["TrainNumber"]: train["Price"] for train in return_transportations}
    
#     total_cost = sum([go_price_map[trans["TrainNumber"]] for trans in planned_transportation[:-1] 
#                      if "TrainNumber" in trans.keys() and trans["TrainNumber"] in go_price_map.keys()])
#     if (len(planned_transportation) > 0 
#         and "TrainNumber" in planned_transportation[-1].keys() 
#         and planned_transportation[-1]["TrainNumber"] in return_price_map.keys()):
#         total_cost += return_price_map[planned_transportation[-1]["TrainNumber"]]
    
#     # 计算住宿费用
#     planned_accommodations = [parse_accommodation_info(item["accommodation"]) 
#                             for item in plan if "accommodation" in item.keys() and item["accommodation"].strip() != "-"]

#     accommodations = json.loads(test_entry["reference_information"])[f"在{dest_city}的酒店"]
#     accommodations = {acc["HotelName"]: acc for acc in accommodations}
    
#     for acc in planned_accommodations:
#         if acc["HotelName"] in accommodations.keys():
#             total_cost += float(accommodations[acc["HotelName"]]["Price"][1:]) * math.ceil(
#                 test_entry["people_number"] / int(
#                     accommodations[acc["HotelName"]]["MaximumOccupancy"]))
    
#     return total_cost

def get_total_cost(test_entry: Dict[str, Any], predicted_entry: Dict[str, Any]) -> float:
    total_cost = 0.0
    org, dest = test_entry["org"], test_entry["dest"]
    people = int(test_entry["people_number"])
    
    # --------------------- 交通费用计算 ---------------------
    for day in predicted_entry["plan"]:
        trans = parse_transport_info(day["transportation"])
        if not trans or trans == "-":
            continue  # 跳过无效交通信息
        
        # 火车费用计算
        if "TrainNumber" in trans:
            # 判断是去程还是返程
            if "从{}到{}".format(org, dest) in trans.get("Route", ""):
                ref_key = f"从{org}到{dest}的列车"
            else:
                ref_key = f"从{dest}到{org}的列车"
            
            # 查找匹配的列车
            trains = json.loads(test_entry["reference_information"]).get(ref_key, [])
            for train in trains:
                if train["TrainNumber"] == trans["TrainNumber"]:
                    price = float(train["Price"].replace("¥", "").strip())
                    total_cost += price * people
                    break
        
        # 航班费用计算（新增完整逻辑）
        elif "FlightNumber" in trans:
            # 判断航班方向
            if f"从{org}到{dest}" in trans.get("Route", ""):
                ref_key = f"从{org}到{dest}的航班"
            else:
                ref_key = f"从{dest}到{org}的航班"
            
            # 查找匹配的航班
            flights = json.loads(test_entry["reference_information"]).get(ref_key, [])
            for flight in flights:
                if flight["FlightNumber"] == trans["FlightNumber"]:
                    # 提取价格（优先取折扣价，若不存在则用原价）
                    price_str = flight.get("DiscountPrice", flight["Price"])
                    price = float(price_str.replace("¥", "").strip())
                    total_cost += price * people
                    break

    # --------------------- 住宿费用计算 ---------------------
    hotels = json.loads(test_entry["reference_information"]).get(f"在{dest}的酒店", [])
    hotel_price_map = {
        acc["HotelName"].split('(')[0].strip(): float(acc["Price"].replace("¥", ""))
        for acc in hotels
    }
    
    # 遍历除最后一天外的所有住宿
    for day in predicted_entry["plan"][:-1]:
        acc_name = parse_accommodation_info(day["accommodation"])
        if acc_name == "-":
            continue
        
        # 模糊匹配酒店名称（允许部分匹配）
        matched_hotels = [name for name in hotel_price_map.keys() if acc_name in name]
        if not matched_hotels:
            continue  # 或根据需求返回错误
        
        # 取第一个匹配酒店的价格
        price = hotel_price_map[matched_hotels[0]]
        max_occupancy = next(
            (int(acc["MaximumOccupancy"]) for acc in hotels 
             if acc_name in acc["HotelName"]),
            2  # 默认最多2人/间
        )
        rooms = math.ceil(people / max_occupancy)
        total_cost += price * rooms
    
    return round(total_cost, 2)  # 保留两位小数

def is_valid_budget(test_entry: Dict[str, Any], predicted_entry: Dict[str, Any]) -> bool:
    budget = test_entry["budget"]
    total_cost = get_total_cost(test_entry, predicted_entry)
    return total_cost <= budget