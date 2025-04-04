# import json
# import math
# import pandas as pd
# from typing import Dict, Any,List

# CSV_FILE_PATH = "/share/home/wuqingyao_zhangboyang/grpo/dataset/3000.csv"
# def load_csv_data() -> pd.DataFrame:
#     """加载CSV文件数据"""
#     return pd.read_csv(CSV_FILE_PATH)

# def create_test_entry(row: pd.Series) -> Dict[str, Any]:
#     """从CSV行创建测试条目"""
#     return {
#         "org": row["org"],
#         "dest": row["dest"],
#         "days": row["days"],
#         "visiting_city_number": row["visiting_city_number"],
#         "date": row["date"],
#         "people_number": row["people_number"],
#         "local_constraint": row["local_constraint"],
#         "budget": row["budget"],
#         "query": row["query"],
#         "level": row["level"],
#         "reference_information": row["reference_information"]
#     }

# def reward_function( prompts,completions:  List[str],**kwargs) -> Dict[int, float]:
#     """
#     计算模型输出的奖励分数，基于CSV文件中的样本数据和模型输出
    
#     参数:
#         csv_file_path: 包含样本数据的CSV文件路径
#         model_outputs: 字典，键为样本索引，值为模型输出字符串
        
#     返回:
#         Dict[int, float]: 字典，键为样本索引，值为对应的奖励分数(0-1)
#     """
#     # 加载CSV数据
#     df = load_csv_data()
#     rewards = {}
    
#     for idx, model_output in enumerate(completions):
#         if idx >= len(df):
#             continue  # 跳过无效索引
            
#         row = df.iloc[idx]
#         test_entry = create_test_entry(row)
        
#         try:
#             # 解析模型输出
#             predicted_entry = json.loads(model_output)
#         except json.JSONDecodeError:
#             rewards[idx] = 0.0  # 无效的JSON格式
#             continue
            
#         # 初始化奖励分数
#         total_score = 0.0
#         max_possible_score = 0.0
        
#         # 1. 检查天数是否有效 (权重: 0.1)
#         max_possible_score += 0.1
#         if is_valid_days(test_entry, {"plan": predicted_entry["travel_plan"]}):
#             total_score += 0.1
        
#         # 2. 检查城市访问顺序是否合理 (权重: 0.1)
#         max_possible_score += 0.1
#         if is_reasonalbe_visiting_city(test_entry, {"plan": predicted_entry["travel_plan"]}):
#             total_score += 0.1
        
#         # 3. 检查景点是否有效 (权重: 0.15)
#         max_possible_score += 0.15
#         if is_valid_attractions(test_entry, {"plan": predicted_entry["travel_plan"]}):
#             total_score += 0.15
        
#         # 4. 检查交通安排是否有效 (权重: 0.2)
#         max_possible_score += 0.2
#         if is_valid_transportation(test_entry, {"plan": predicted_entry["travel_plan"]}):
#             total_score += 0.2
        
#         # 5. 检查餐厅是否有效且不重复 (权重: 0.15)
#         max_possible_score += 0.15
#         if is_valid_restaurants(test_entry, {"plan": predicted_entry["travel_plan"]}):
#             total_score += 0.15
        
#         # 6. 检查住宿是否有效 (权重: 0.1)
#         max_possible_score += 0.1
#         if is_valid_accommodation(test_entry, {"plan": predicted_entry["travel_plan"]}):
#             total_score += 0.1
        
#         # 7. 检查预算是否合理 (权重: 0.2)
#         max_possible_score += 0.2
#         if is_valid_budget(test_entry, {"plan": predicted_entry["travel_plan"]}):
#             total_score += 0.2
        
#         # 计算最终奖励分数 (归一化到0-1)
#         final_score = total_score / max_possible_score if max_possible_score > 0 else 0
#         rewards[idx] = final_score
    
#     return rewards

# # 以下是评估脚本中的辅助函数，保持不变
# def is_valid_days(test_entry: Dict[str, Any], predicted_entry: Dict[str, Any]) -> bool:
#     if test_entry["days"] != len(predicted_entry["plan"]):
#         return False
#     return True


# def is_reasonalbe_visiting_city(test_entry: Dict[str, Any], predicted_entry: Dict[str, Any]) -> bool:
#     plan = predicted_entry["plan"]
#     cities = [item["current_city"] for item in plan]
    
#     org, dest = test_entry["org"], test_entry["dest"]
#     expected_start = f"从{org}到{dest}"
#     expected_end = f"从{dest}到{org}"
    
#     # 检查首尾两天
#     if cities[0] != expected_start or cities[-1] != expected_end:
#         return False
    
#     # 检查中间天数是否全部为目的地
#     for city in cities[1:-1]:
#         if city != dest:
#             return False
    
#     return True

# # def is_valid_attractions(test_entry: Dict[str, Any], predicted_entry: Dict[str, Any]) -> bool:
# #     plan = predicted_entry["plan"]
# #     planned_attractions = [item["attraction"].split(",")[0] 
# #                           for item in plan if item["attraction"] != "-"]
    
# #     dest_city = test_entry["dest"]
# #     attractions = json.loads(test_entry["reference_information"])[f"在{dest_city}的景点"]
# #     attractions = [attr["Name"] for attr in attractions]
    
# #     for attr in planned_attractions:
# #         if attr not in attractions:
# #             return False
# #     return True

# def is_valid_attractions(test_entry: Dict[str, Any], predicted_entry: Dict[str, Any]) -> bool:
#     plan = predicted_entry["plan"]
#     dest = test_entry["dest"]
#     reference_attractions = json.loads(test_entry["reference_information"])[f"在{dest}的景点"]
#     valid_attractions = {attr["Name"].split('(')[0].strip() for attr in reference_attractions}
    
#     for day in plan:
#         # 确保attraction是字符串类型
#         attractions_str = day["attraction"] if isinstance(day["attraction"], str) else ", ".join(day["attraction"])
#         attractions = [attr.split(',')[0].split('(')[0].strip() 
#                       for attr in attractions_str.split(', ') if attr != "-"]
#         for attr in attractions:
#             if attr not in valid_attractions:
#                 return False
#     return True

# def is_valid_restaurants(test_entry: Dict[str, Any], predicted_entry: Dict[str, Any]) -> bool:
#     restaurants_list = []
#     plan = predicted_entry["plan"]
    
#     for i, unit in enumerate(plan):
#         # 处理lunch
#         lunch = unit.get('lunch', '-')
#         if isinstance(lunch, list):
#             lunch = lunch[0] if lunch else '-'
#         if lunch and lunch != '-':
#             if lunch not in restaurants_list:
#                 restaurants_list.append(lunch)
#             else:
#                 return False
        
#         # 处理dinner
#         dinner = unit.get('dinner', '-')
#         if isinstance(dinner, list):
#             dinner = dinner[0] if dinner else '-'
#         if dinner and dinner != '-':
#             if dinner not in restaurants_list:
#                 restaurants_list.append(dinner)
#             else:
#                 return False
    
#     return True



# def parse_transport_info(info_string: str) -> Dict[str, Any]:
#     parts = [part.strip() for part in info_string.split(',')]
#     transport_info = {}
#     route = []  # 新增：提取路线信息
    
#     for part in parts:
#         if ':' in part:
#             key, value = part.split(':', 1)
#             key, value = key.strip(), value.strip()
#             if key == "TrainNumber":
#                 value = value.split(',')[0].strip()
#             transport_info[key] = value
#         elif "到" in part:  # 捕获路线信息（如"从青岛到秦皇岛"）
#             route.append(part)
    
#     if route:
#         transport_info["Route"] = route[0]  # 记录路线信息
    
#     return transport_info

# def is_valid_transportation(test_entry: Dict[str, Any], predicted_entry: Dict[str, Any]) -> bool:
#     plan = predicted_entry["plan"]
#     planned_transportation = [parse_transport_info(item["transportation"]) 
#                              if item["transportation"].strip() != "-" else "-"
#                              for item in plan]
    
#     org_city = test_entry["org"]
#     dest_city = test_entry["dest"]

#     # 获取火车的参考信息
#     go_transportations = json.loads(test_entry["reference_information"])[f"从{org_city}到{dest_city}的列车"]
#     go_train_num = [train["TrainNumber"] for train in go_transportations]
#     go_departure_time = [train["DepartureTime"] for train in go_transportations]
#     go_arrival_time = [train["ArrivalTime"] for train in go_transportations]
    
#     return_transportations = json.loads(test_entry["reference_information"])[f"从{dest_city}到{org_city}的列车"]
#     return_train_num = [train["TrainNumber"] for train in return_transportations]
#     return_departure_time = [train["DepartureTime"] for train in return_transportations]
#     return_arrival_time = [train["ArrivalTime"] for train in return_transportations]

#     # 获取航班的参考信息
#     go_flight = json.loads(test_entry["reference_information"])[f"从{org_city}到{dest_city}的航班"]
#     go_flights_number = [flight["FlightNumber"] for flight in go_flight]
#     go_flight_departure_time = [flight["DepTime"] for flight in go_flight]
#     go_flight_arrival_time = [flight["ArrTime"] for flight in go_flight]

#     return_flight = json.loads(test_entry["reference_information"])[f"从{dest_city}到{org_city}的航班"]
#     return_flights_number = [flight["FlightNumber"] for flight in return_flight]
#     return_flight_departure_time = [flight["DepTime"] for flight in return_flight]
#     return_flight_arrival_time = [flight["ArrTime"] for flight in return_flight]
    
#     try:
#         for i, trans in enumerate(planned_transportation):
#             if "TrainNumber" in trans:
#                 if i == 0:
#                     if trans == "-":
#                         return False
#                     if (trans["TrainNumber"] not in go_train_num
#                         or trans["DepartureTime"] not in go_departure_time
#                         or trans["ArrivalTime"] not in go_arrival_time):
#                         return False
#                 elif i == len(planned_transportation) - 1:
#                     if trans == "-":
#                         return False
#                     if (trans["TrainNumber"] not in return_train_num
#                         or trans["DepartureTime"] not in return_departure_time
#                         or trans["ArrivalTime"] not in return_arrival_time):
#                         return False
#                 else:
#                     if trans != "-":
#                         return False
                    
#             elif "FlightNumber" in trans:
#                 if i == 0:
#                     if trans == "-":
#                         return False
#                     if(trans["FlightNumber"] not in go_flights_number
#                        or trans["DepTime"] not in go_flight_departure_time
#                        or trans["ArrTime"] not in go_flight_arrival_time):
#                         return False
#                 elif i == len(planned_transportation) - 1:
#                     if trans == "-":
#                         return False
#                     if (trans["FlightNumber"] not in return_flights_number
#                        or trans["DepTime"] not in return_flight_departure_time
#                        or trans["ArrTime"] not in return_flight_arrival_time):
#                         return False
#                 else:
#                     if trans != "-":
#                         return False
    
#     except Exception as e:
#         return False
#     return True

# # def parse_accommodation_info(info_string: str) -> Dict[str, Any]:
# #     if info_string.strip() == "-":
# #         return "-"
    
# #     parts = [part.strip() for part in info_string.split(',')]
# #     if len(parts) == 1:
# #         return {"HotelName": parts[0], "HouseRules": None}
# #     return {"HotelName": parts[0], "HouseRules": parts[1]}

# def parse_accommodation_info(info_string: str) -> str:
#     """直接返回酒店名称字符串，忽略房规信息"""
#     if info_string.strip() == "-":
#         return "-"
#     # 提取酒店名称（假设名称在逗号或括号前）
#     name = info_string.split(',')[0].split('(')[0].strip()
#     return name



# def is_valid_accommodation(test_entry: Dict[str, Any], predicted_entry: Dict[str, Any]) -> bool:
#     plan = predicted_entry["plan"]
#     accommodations = json.loads(test_entry["reference_information"])[f"在{test_entry['dest']}的酒店"]
#     valid_hotels = {acc["HotelName"].split('(')[0].strip() for acc in accommodations}  # 统一处理名称格式
    
#     for i, day_plan in enumerate(plan):
#         acc_name = parse_accommodation_info(day_plan["accommodation"])
#         # 最后一天允许无住宿，其他天必须有效
#         if i != len(plan)-1 and (acc_name == "-" or acc_name not in valid_hotels):
#             return False
#     return True



# def get_total_cost(test_entry: Dict[str, Any], predicted_entry: Dict[str, Any]) -> float:
#     total_cost = 0.0
#     org, dest = test_entry["org"], test_entry["dest"]
#     people = int(test_entry["people_number"])
    
#     # --------------------- 交通费用计算 ---------------------
#     for day in predicted_entry["plan"]:
#         trans = parse_transport_info(day["transportation"])
#         if not trans or trans == "-":
#             continue  # 跳过无效交通信息
        
#         # 火车费用计算
#         if "TrainNumber" in trans:
#             # 判断是去程还是返程
#             if "从{}到{}".format(org, dest) in trans.get("Route", ""):
#                 ref_key = f"从{org}到{dest}的列车"
#             else:
#                 ref_key = f"从{dest}到{org}的列车"
            
#             # 查找匹配的列车
#             trains = json.loads(test_entry["reference_information"]).get(ref_key, [])
#             for train in trains:
#                 if train["TrainNumber"] == trans["TrainNumber"]:
#                     price = float(train["Price"].replace("¥", "").strip())
#                     total_cost += price * people
#                     break
        
#         # 航班费用计算（新增完整逻辑）
#         elif "FlightNumber" in trans:
#             # 判断航班方向
#             if f"从{org}到{dest}" in trans.get("Route", ""):
#                 ref_key = f"从{org}到{dest}的航班"
#             else:
#                 ref_key = f"从{dest}到{org}的航班"
            
#             # 查找匹配的航班
#             flights = json.loads(test_entry["reference_information"]).get(ref_key, [])
#             for flight in flights:
#                 if flight["FlightNumber"] == trans["FlightNumber"]:
#                     # 提取价格（优先取折扣价，若不存在则用原价）
#                     price_str = flight.get("DiscountPrice", flight["Price"])
#                     price = float(price_str.replace("¥", "").strip())
#                     total_cost += price * people
#                     break

#     # --------------------- 住宿费用计算 ---------------------
#     hotels = json.loads(test_entry["reference_information"]).get(f"在{dest}的酒店", [])
#     hotel_price_map = {
#         acc["HotelName"].split('(')[0].strip(): float(acc["Price"].replace("¥", ""))
#         for acc in hotels
#     }
    
#     # 遍历除最后一天外的所有住宿
#     for day in predicted_entry["plan"][:-1]:
#         acc_name = parse_accommodation_info(day["accommodation"])
#         if acc_name == "-":
#             continue
        
#         # 模糊匹配酒店名称（允许部分匹配）
#         matched_hotels = [name for name in hotel_price_map.keys() if acc_name in name]
#         if not matched_hotels:
#             continue  # 或根据需求返回错误
        
#         # 取第一个匹配酒店的价格
#         price = hotel_price_map[matched_hotels[0]]
#         max_occupancy = next(
#             (int(acc["MaximumOccupancy"]) for acc in hotels 
#              if acc_name in acc["HotelName"]),
#             2  # 默认最多2人/间
#         )
#         rooms = math.ceil(people / max_occupancy)
#         total_cost += price * rooms
    
#     return round(total_cost, 2)  # 保留两位小数

# def is_valid_budget(test_entry: Dict[str, Any], predicted_entry: Dict[str, Any]) -> bool:
#     budget = test_entry["budget"]
#     total_cost = get_total_cost(test_entry, predicted_entry)
#     return total_cost <= budget

import json
import math
import pandas as pd
from typing import Dict, Any, List

CSV_FILE_PATH = "/share/home/wuqingyao_zhangboyang/grpo/dataset/3000.csv"

def load_csv_data() -> pd.DataFrame:
    """Load CSV file data"""
    return pd.read_csv(CSV_FILE_PATH)

def create_test_entry(row: pd.Series) -> Dict[str, Any]:
    """Create test entry from CSV row"""
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

def get_departure_time(transport_info: Dict[str, Any]) -> str:
    """Get departure time handling both DepTime and DepartureTime fields"""
    return transport_info.get("DepartureTime") or transport_info.get("DepTime") or ""

def get_arrival_time(transport_info: Dict[str, Any]) -> str:
    """Get arrival time handling both ArrivalTime and ArrTime fields"""
    return transport_info.get("ArrivalTime") or transport_info.get("ArrTime") or ""

def parse_transport_info(info_string: str) -> Dict[str, Any]:
    """Parse transportation information string into a dictionary"""
    if info_string.strip() == "-":
        return "-"
    
    parts = [part.strip() for part in info_string.split(',')]
    transport_info = {}
    route = []  # For capturing route information
    
    for part in parts:
        if ':' in part:
            key, value = part.split(':', 1)
            key, value = key.strip(), value.strip()
            if key == "TrainNumber":
                value = value.split(',')[0].strip()
            transport_info[key] = value
        elif "到" in part:  # Capture route information
            route.append(part)
    
    if route:
        transport_info["Route"] = route[0]
    
    return transport_info

def parse_accommodation_info(info_string: str) -> str:
    """Extract hotel name, ignoring house rules"""
    if info_string.strip() == "-":
        return "-"
    # Extract hotel name (assuming name is before comma or parenthesis)
    name = info_string.split(',')[0].split('(')[0].strip()
    return name

def is_valid_days(test_entry: Dict[str, Any], predicted_entry: Dict[str, Any]) -> bool:
    """Validate number of days matches"""
    return test_entry["days"] == len(predicted_entry["plan"])

def is_reasonalbe_visiting_city(test_entry: Dict[str, Any], predicted_entry: Dict[str, Any]) -> bool:
    """Validate city visiting sequence is reasonable"""
    plan = predicted_entry["plan"]
    org, dest = test_entry["org"], test_entry["dest"]
    expected_start = f"从{org}到{dest}"
    expected_end = f"从{dest}到{org}"
    
    # Check first and last day
    if plan[0]["current_city"] != expected_start or plan[-1]["current_city"] != expected_end:
        return False
    
    # Check middle days are all at destination
    for day in plan[1:-1]:
        if day["current_city"] != dest:
            return False
    
    return True

def is_valid_attractions(test_entry: Dict[str, Any], predicted_entry: Dict[str, Any]) -> bool:
    """Validate attractions in the plan"""
    plan = predicted_entry["plan"]
    dest = test_entry["dest"]
    
    try:
        reference_attractions = json.loads(test_entry["reference_information"])[f"在{dest}的景点"]
        valid_attractions = {attr["Name"].split('(')[0].strip() for attr in reference_attractions}
    except (KeyError, json.JSONDecodeError):
        return False
    
    for day in plan:
        attractions_str = day["attraction"] if isinstance(day["attraction"], str) else ", ".join(day["attraction"])
        attractions = [attr.split(',')[0].split('(')[0].strip() 
                      for attr in attractions_str.split(', ') if attr != "-"]
        for attr in attractions:
            if attr not in valid_attractions:
                return False
    return True

def is_valid_restaurants(test_entry: Dict[str, Any], predicted_entry: Dict[str, Any]) -> bool:
    """Validate restaurants in the plan"""
    restaurants_list = []
    plan = predicted_entry["plan"]
    
    try:
        reference_restaurants = json.loads(test_entry["reference_information"])[f"在{test_entry['dest']}的餐厅"]
        valid_restaurants = {rest["Name"] for rest in reference_restaurants}
    except (KeyError, json.JSONDecodeError):
        return False
    
    for day in plan:
        # Handle lunch
        lunch = day.get('lunch', '-')
        if isinstance(lunch, list):
            lunch = lunch[0] if lunch else '-'
        if lunch and lunch != '-':
            if lunch not in valid_restaurants:
                return False
            if lunch in restaurants_list:
                return False
            restaurants_list.append(lunch)
        
        # Handle dinner
        dinner = day.get('dinner', '-')
        if isinstance(dinner, list):
            dinner = dinner[0] if dinner else '-'
        if dinner and dinner != '-':
            if dinner not in valid_restaurants:
                return False
            if dinner in restaurants_list:
                return False
            restaurants_list.append(dinner)
    
    return True

def is_valid_transportation(test_entry: Dict[str, Any], predicted_entry: Dict[str, Any]) -> bool:
    """Validate all transportation in the plan"""
    plan = predicted_entry["plan"]
    org, dest = test_entry["org"], test_entry["dest"]
    
    try:
        ref_info = json.loads(test_entry["reference_information"])
    except json.JSONDecodeError:
        return False
    
    # Check transportation constraints
    local_constraint = test_entry.get("local_constraint", "")
    constraints = parse_local_constraint(local_constraint)
    transportation_constraint = constraints.get("transportation", None)
    
    # Validate each day's transportation
    for i, day in enumerate(plan):
        transport_info = day.get("transportation", "-").strip()
        
        # Skip empty transportation for intermediate days
        if 0 < i < len(plan)-1:
            if transport_info != "-":
                return False
            continue
            
        # Parse transportation data
        transport_data = parse_transport_info(transport_info)
        if transport_data == "-":
            return False
            
        # Check transportation type constraint
        if transportation_constraint:
            transport_type = None
            if "TrainNumber" in transport_data:
                transport_type = "坐火车"
            elif "FlightNumber" in transport_data:
                transport_type = "坐飞机"
                
            if transport_type:
                if (transportation_constraint == "不要坐火车" and transport_type == "坐火车") or \
                   (transportation_constraint == "不要坐飞机" and transport_type == "坐飞机"):
                    return False
                elif (transportation_constraint == "坐火车" and transport_type != "坐火车") or \
                     (transportation_constraint == "坐飞机" and transport_type != "坐飞机"):
                    return False
        
        # Validate specific transportation
        if i == 0:  # Outbound trip
            if not is_valid_go_transport(transport_info, test_entry):
                return False
        elif i == len(plan)-1:  # Return trip
            if not is_valid_return_transport(transport_info, test_entry):
                return False
                
    return True

def is_valid_go_transport(transport_info: str, test_entry: dict) -> bool:
    """Validate outbound transportation"""
    if transport_info == "-":
        return False
        
    org, dest = test_entry["org"], test_entry["dest"]
    transport_data = parse_transport_info(transport_info)
    
    try:
        ref_info = json.loads(test_entry["reference_information"])
        
        if "TrainNumber" in transport_data:
            go_transportations = ref_info.get(f"从{org}到{dest}的列车", [])
            for train in go_transportations:
                if (train["TrainNumber"] == transport_data["TrainNumber"] and
                    get_departure_time(train) == get_departure_time(transport_data) and
                    get_arrival_time(train) == get_arrival_time(transport_data)):
                    return True
        elif "FlightNumber" in transport_data:
            go_flights = ref_info.get(f"从{org}到{dest}的航班", [])
            for flight in go_flights:
                if (flight["FlightNumber"] == transport_data["FlightNumber"] and
                    get_departure_time(flight) == get_departure_time(transport_data) and
                    get_arrival_time(flight) == get_arrival_time(transport_data)):
                    return True
    except (KeyError, TypeError, json.JSONDecodeError):
        pass
        
    return False

def is_valid_return_transport(transport_info: str, test_entry: dict) -> bool:
    """Validate return transportation"""
    if transport_info == "-":
        return False
        
    org, dest = test_entry["org"], test_entry["dest"]
    transport_data = parse_transport_info(transport_info)
    
    try:
        ref_info = json.loads(test_entry["reference_information"])
        
        if "TrainNumber" in transport_data:
            return_transportations = ref_info.get(f"从{dest}到{org}的列车", [])
            for train in return_transportations:
                if (train["TrainNumber"] == transport_data["TrainNumber"] and
                    get_departure_time(train) == get_departure_time(transport_data) and
                    get_arrival_time(train) == get_arrival_time(transport_data)):
                    return True
        elif "FlightNumber" in transport_data:
            return_flights = ref_info.get(f"从{dest}到{org}的航班", [])
            for flight in return_flights:
                if (flight["FlightNumber"] == transport_data["FlightNumber"] and
                    get_departure_time(flight) == get_departure_time(transport_data) and
                    get_arrival_time(flight) == get_arrival_time(transport_data)):
                    return True
    except (KeyError, TypeError, json.JSONDecodeError):
        pass
        
    return False

def is_valid_accommodation(test_entry: Dict[str, Any], predicted_entry: Dict[str, Any]) -> bool:
    """Validate accommodation information"""
    plan = predicted_entry["plan"]
    dest = test_entry["dest"]
    
    try:
        accommodations = json.loads(test_entry["reference_information"])[f"在{dest}的酒店"]
        valid_hotels = {acc["HotelName"].split('(')[0].strip() for acc in accommodations}
    except (KeyError, json.JSONDecodeError):
        return False
        
    for i, day in enumerate(plan):
        acc_info = day.get("accommodation", "-").strip()
        if i != len(plan)-1 and (acc_info == "-" or parse_accommodation_info(acc_info) not in valid_hotels):
            return False
            
    return True

def get_total_cost(test_entry: Dict[str, Any], predicted_entry: Dict[str, Any]) -> float:
    """Calculate total cost of the trip"""
    total_cost = 0.0
    org, dest = test_entry["org"], test_entry["dest"]
    people = int(test_entry["people_number"])
    
    try:
        ref_info = json.loads(test_entry["reference_information"])
    except json.JSONDecodeError:
        return float('inf')  # Return high cost if reference info is invalid
    
    # Transportation costs
    for i, day in enumerate(predicted_entry["plan"]):
        transport_info = day.get("transportation", "-").strip()
        if transport_info == "-":
            continue
            
        transport_data = parse_transport_info(transport_info)
        if transport_data == "-":
            continue
            
        if i == 0:  # Outbound trip
            if "TrainNumber" in transport_data:
                trains = ref_info.get(f"从{org}到{dest}的列车", [])
                for train in trains:
                    if (train["TrainNumber"] == transport_data["TrainNumber"] and
                        get_departure_time(train) == get_departure_time(transport_data)):
                        total_cost += float(train["Price"]) * people
                        break
            elif "FlightNumber" in transport_data:
                flights = ref_info.get(f"从{org}到{dest}的航班", [])
                for flight in flights:
                    if (flight["FlightNumber"] == transport_data["FlightNumber"] and
                        get_departure_time(flight) == get_departure_time(transport_data)):
                        price_str = flight.get("DiscountPrice", flight["Price"])
                        total_cost += float(price_str.replace("¥", "").strip()) * people
                        break
        elif i == len(predicted_entry["plan"])-1:  # Return trip
            if "TrainNumber" in transport_data:
                trains = ref_info.get(f"从{dest}到{org}的列车", [])
                for train in trains:
                    if (train["TrainNumber"] == transport_data["TrainNumber"] and
                        get_departure_time(train) == get_departure_time(transport_data)):
                        total_cost += float(train["Price"]) * people
                        break
            elif "FlightNumber" in transport_data:
                flights = ref_info.get(f"从{dest}到{org}的航班", [])
                for flight in flights:
                    if (flight["FlightNumber"] == transport_data["FlightNumber"] and
                        get_departure_time(flight) == get_departure_time(transport_data)):
                        price_str = flight.get("DiscountPrice", flight["Price"])
                        total_cost += float(price_str.replace("¥", "").strip()) * people
                        break
    
    # Accommodation costs
    try:
        hotels = ref_info.get(f"在{dest}的酒店", [])
        hotel_price_map = {
            acc["HotelName"].split('(')[0].strip(): float(acc["Price"].replace("¥", ""))
            for acc in hotels
        }
        hotel_capacity_map = {
            acc["HotelName"].split('(')[0].strip(): int(acc["MaximumOccupancy"])
            for acc in hotels
        }
    except (KeyError, ValueError):
        hotel_price_map = {}
        hotel_capacity_map = {}
    
    # Calculate accommodation costs for all days except last
    for day in predicted_entry["plan"][:-1]:
        acc_name = parse_accommodation_info(day.get("accommodation", "-"))
        if acc_name == "-":
            continue
            
        # Find matching hotel
        matched_hotels = [name for name in hotel_price_map.keys() if acc_name in name]
        if matched_hotels:
            price = hotel_price_map[matched_hotels[0]]
            capacity = hotel_capacity_map.get(matched_hotels[0], 2)  # Default to 2 if not found
            rooms = math.ceil(people / capacity)
            total_cost += price * rooms
    
    return round(total_cost, 2)

def is_valid_budget(test_entry: Dict[str, Any], predicted_entry: Dict[str, Any]) -> bool:
    """Validate if total cost is within budget"""
    try:
        budget = float(test_entry["budget"])
        total_cost = get_total_cost(test_entry, predicted_entry)
        return total_cost <= budget
    except (KeyError, ValueError):
        return False

def parse_local_constraint(local_constraint: str) -> Dict[str, str]:
    """Parse local constraint string into a dictionary"""
    constraints = {}
    if not local_constraint:
        return constraints
        
    for item in local_constraint.split(","):
        if ':' in item:
            key, value = item.split(":", 1)
            constraints[key.strip()] = value.strip()
    return constraints

def reward_function(prompts, completions: List[str], **kwargs) -> Dict[int, float]:
    """
    Calculate reward scores for model outputs based on test data
    
    Args:
        prompts: List of prompts (unused in this function)
        completions: List of model output strings
        **kwargs: Additional arguments
        
    Returns:
        Dict[int, float]: Dictionary with index as key and reward score (0-1) as value
    """
    df = load_csv_data()
    rewards = {}
    
    for idx, model_output in enumerate(completions):
        if idx >= len(df):
            continue  # Skip invalid indices
            
        row = df.iloc[idx]
        test_entry = create_test_entry(row)
        
        try:
            predicted_entry = json.loads(model_output)
            travel_plan = predicted_entry.get("travel_plan", [])
        except json.JSONDecodeError:
            rewards[idx] = 0.0  # Invalid JSON format
            continue
            
        # Initialize scoring
        score_weights = {
            'days': 0.1,
            'city_sequence': 0.1,
            'attractions': 0.15,
            'transportation': 0.2,
            'restaurants': 0.15,
            'accommodation': 0.1,
            'budget': 0.2
        }
        total_score = 0.0
        
        # Create plan dictionary for validation functions
        plan_dict = {"plan": travel_plan}
        
        # 1. Validate days
        if is_valid_days(test_entry, plan_dict):
            total_score += score_weights['days']
        
        # 2. Validate city sequence
        if is_reasonalbe_visiting_city(test_entry, plan_dict):
            total_score += score_weights['city_sequence']
        
        # 3. Validate attractions
        if is_valid_attractions(test_entry, plan_dict):
            total_score += score_weights['attractions']
        
        # 4. Validate transportation
        if is_valid_transportation(test_entry, plan_dict):
            total_score += score_weights['transportation']
        
        # 5. Validate restaurants
        if is_valid_restaurants(test_entry, plan_dict):
            total_score += score_weights['restaurants']
        
        # 6. Validate accommodation
        if is_valid_accommodation(test_entry, plan_dict):
            total_score += score_weights['accommodation']
        
        # 7. Validate budget
        if is_valid_budget(test_entry, plan_dict):
            total_score += score_weights['budget']
        
        # Normalize score to 0-1 range
        rewards[idx] = total_score / sum(score_weights.values())
    
    return rewards