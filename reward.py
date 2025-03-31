import json
import math
import pandas as pd
from typing import Dict, Any, List

CSV_FILE_PATH = "/share/home/wuqingyao_zhangboyang/grpo/n_dataset/n_train.csv"

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
    if not isinstance(info_string, str) or info_string.strip() == "-":
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
    if not isinstance(info_string, str) or info_string.strip() == "-":
        return "-"
    # Extract hotel name (assuming name is before comma or parenthesis)
    name = info_string.split(',')[0].split('(')[0].strip()
    return name

def is_valid_days(test_entry: Dict[str, Any], predicted_entry: Dict[str, Any]) -> bool:
    """Validate number of days matches"""
    if not predicted_entry.get("plan"):
        return False
    return test_entry["days"] == len(predicted_entry["plan"])

def is_reasonalbe_visiting_city(test_entry: Dict[str, Any], predicted_entry: Dict[str, Any]) -> bool:
    """Validate city visiting sequence is reasonable"""
    if not predicted_entry.get("plan"):
        return False
        
    plan = predicted_entry["plan"]
    if not plan:  # Empty plan
        return False
        
    org, dest = test_entry["org"], test_entry["dest"]
    expected_start = f"从{org}到{dest}"
    expected_end = f"从{dest}到{org}"
    
    try:
        # Check first and last day
        if plan[0].get("current_city") != expected_start or plan[-1].get("current_city") != expected_end:
            return False
        
        # Check middle days are all at destination
        for day in plan[1:-1]:
            if day.get("current_city") != dest:
                return False
    except (KeyError, IndexError):
        return False
    
    return True

def is_valid_attractions(test_entry: Dict[str, Any], predicted_entry: Dict[str, Any]) -> bool:
    """Validate attractions in the plan"""
    if not predicted_entry.get("plan"):
        return False
        
    plan = predicted_entry["plan"]
    dest = test_entry["dest"]
    
    try:
        reference_attractions = json.loads(test_entry["reference_information"])[f"在{dest}的景点"]
        valid_attractions = {attr["Name"].split('(')[0].strip() for attr in reference_attractions}
    except (KeyError, json.JSONDecodeError, AttributeError):
        return False
    
    for day in plan:
        try:
            attractions_str = day["attraction"] if isinstance(day["attraction"], str) else ", ".join(day["attraction"])
            attractions = [attr.split(',')[0].split('(')[0].strip() 
                          for attr in attractions_str.split(', ') if attr != "-"]
            for attr in attractions:
                if attr not in valid_attractions:
                    return False
        except (KeyError, AttributeError):
            return False
    return True

def is_valid_restaurants(test_entry: Dict[str, Any], predicted_entry: Dict[str, Any]) -> bool:
    """Validate restaurants in the plan"""
    if not predicted_entry.get("plan"):
        return False
        
    restaurants_list = []
    plan = predicted_entry["plan"]
    
    try:
        reference_restaurants = json.loads(test_entry["reference_information"])[f"在{test_entry['dest']}的餐厅"]
        valid_restaurants = {rest["Name"] for rest in reference_restaurants}
    except (KeyError, json.JSONDecodeError, AttributeError):
        return False
    
    for day in plan:
        try:
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
        except (KeyError, AttributeError):
            return False
    
    return True

def is_valid_transportation(test_entry: Dict[str, Any], predicted_entry: Dict[str, Any]) -> bool:
    """Validate all transportation in the plan"""
    if not predicted_entry.get("plan"):
        return False
        
    plan = predicted_entry["plan"]
    org, dest = test_entry["org"], test_entry["dest"]
    
    try:
        ref_info = json.loads(test_entry["reference_information"])
    except (json.JSONDecodeError, AttributeError):
        return False
    
    # Check transportation constraints
    local_constraint = test_entry.get("local_constraint", "")
    constraints = parse_local_constraint(local_constraint)
    transportation_constraint = constraints.get("transportation", None)
    
    # Validate each day's transportation
    for i, day in enumerate(plan):
        try:
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
        except (KeyError, AttributeError):
            return False
                
    return True

def is_valid_go_transport(transport_info: str, test_entry: dict) -> bool:
    """Validate outbound transportation"""
    if not isinstance(transport_info, str) or transport_info == "-":
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
    except (KeyError, TypeError, json.JSONDecodeError, AttributeError):
        pass
        
    return False

def is_valid_return_transport(transport_info: str, test_entry: dict) -> bool:
    """Validate return transportation"""
    if not isinstance(transport_info, str) or transport_info == "-":
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
    except (KeyError, TypeError, json.JSONDecodeError, AttributeError):
        pass
        
    return False

def is_valid_accommodation(test_entry: Dict[str, Any], predicted_entry: Dict[str, Any]) -> bool:
    """Validate accommodation information"""
    if not predicted_entry.get("plan"):
        return False
        
    plan = predicted_entry["plan"]
    dest = test_entry["dest"]
    
    try:
        accommodations = json.loads(test_entry["reference_information"])[f"在{dest}的酒店"]
        valid_hotels = {acc["HotelName"].split('(')[0].strip() for acc in accommodations}
    except (KeyError, json.JSONDecodeError, AttributeError):
        return False
        
    for i, day in enumerate(plan):
        try:
            acc_info = day.get("accommodation", "-").strip()
            if i != len(plan)-1 and (acc_info == "-" or parse_accommodation_info(acc_info) not in valid_hotels):
                return False
        except (KeyError, AttributeError):
            return False
            
    return True

# def get_total_cost(test_entry: Dict[str, Any], predicted_entry: Dict[str, Any]) -> float:
#     """Calculate total cost of the trip"""
#     if not predicted_entry.get("plan"):
#         return float('inf')  # Return high cost if plan is invalid
        
#     total_cost = 0.0
#     org, dest = test_entry["org"], test_entry["dest"]
#     people = int(test_entry["people_number"])
    
#     try:
#         ref_info = json.loads(test_entry["reference_information"])
#     except (json.JSONDecodeError, AttributeError):
#         return float('inf')  # Return high cost if reference info is invalid
    
#     # Transportation costs
#     for i, day in enumerate(predicted_entry["plan"]):
#         try:
#             transport_info = day.get("transportation", "-").strip()
#             if transport_info == "-":
#                 continue
                
#             transport_data = parse_transport_info(transport_info)
#             if transport_data == "-":
#                 continue
                
#             if i == 0:  # Outbound trip
#                 if "TrainNumber" in transport_data:
#                     trains = ref_info.get(f"从{org}到{dest}的列车", [])
#                     for train in trains:
#                         if (train["TrainNumber"] == transport_data["TrainNumber"] and
#                             get_departure_time(train) == get_departure_time(transport_data)):
#                             total_cost += float(train["Price"]) * people
#                             break
#                 elif "FlightNumber" in transport_data:
#                     flights = ref_info.get(f"从{org}到{dest}的航班", [])
#                     for flight in flights:
#                         if (flight["FlightNumber"] == transport_data["FlightNumber"] and
#                             get_departure_time(flight) == get_departure_time(transport_data)):
#                             price_str = flight.get("DiscountPrice", flight["Price"])
#                             total_cost += float(price_str.replace("¥", "").strip()) * people
#                             break
#             elif i == len(predicted_entry["plan"])-1:  # Return trip
#                 if "TrainNumber" in transport_data:
#                     trains = ref_info.get(f"从{dest}到{org}的列车", [])
#                     for train in trains:
#                         if (train["TrainNumber"] == transport_data["TrainNumber"] and
#                             get_departure_time(train) == get_departure_time(transport_data)):
#                             total_cost += float(train["Price"]) * people
#                             break
#                 elif "FlightNumber" in transport_data:
#                     flights = ref_info.get(f"从{dest}到{org}的航班", [])
#                     for flight in flights:
#                         if (flight["FlightNumber"] == transport_data["FlightNumber"] and
#                             get_departure_time(flight) == get_departure_time(transport_data)):
#                             price_str = flight.get("DiscountPrice", flight["Price"])
#                             total_cost += float(price_str.replace("¥", "").strip()) * people
#                             break
#         except (KeyError, ValueError, AttributeError):
#             continue
    
#     # Accommodation costs
#     try:
#         hotels = ref_info.get(f"在{dest}的酒店", [])
#         hotel_price_map = {
#             acc["HotelName"].split('(')[0].strip(): float(acc["Price"].replace("¥", ""))
#             for acc in hotels
#         }
#         hotel_capacity_map = {
#             acc["HotelName"].split('(')[0].strip(): int(acc["MaximumOccupancy"])
#             for acc in hotels
#         }
#     except (KeyError, ValueError, AttributeError):
#         hotel_price_map = {}
#         hotel_capacity_map = {}
    
#     # Calculate accommodation costs for all days except last
#     for day in predicted_entry["plan"][:-1]:
#         try:
#             acc_name = parse_accommodation_info(day.get("accommodation", "-"))
#             if acc_name == "-":
#                 continue
                
#             # Find matching hotel
#             matched_hotels = [name for name in hotel_price_map.keys() if acc_name in name]
#             if matched_hotels:
#                 price = hotel_price_map[matched_hotels[0]]
#                 capacity = hotel_capacity_map.get(matched_hotels[0], 2)  # Default to 2 if not found
#                 rooms = math.ceil(people / capacity)
#                 total_cost += price * rooms
#         except (KeyError, ValueError, AttributeError):
#             continue
    
#     return round(total_cost, 2)
def get_total_cost(test_entry: Dict[str, Any], predicted_entry: Dict[str, Any]) -> float:
    """Calculate total cost of the trip (交通 + 住宿 + 餐饮)"""
    if not predicted_entry.get("plan"):
        return float('inf')

    total_cost = 0.0
    org, dest = test_entry["org"], test_entry["dest"]
    people = int(test_entry["people_number"])

    try:
        ref_info = json.loads(test_entry["reference_information"])
    except (json.JSONDecodeError, AttributeError):
        return float('inf')

    # ---------------------- 1. 计算交通费用 ----------------------
    for i, day in enumerate(predicted_entry["plan"]):
        transport_info = day.get("transportation", "-").strip()
        if transport_info == "-":
            continue

        transport_data = parse_transport_info(transport_info)
        if transport_data == "-":
            continue

        # 去程交通（第一天）
        if i == 0:
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

        # 返程交通（最后一天）
        elif i == len(predicted_entry["plan"]) - 1:
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

    # ---------------------- 2. 计算住宿费用 ----------------------
    hotels = ref_info.get(f"在{dest}的酒店", [])
    hotel_price_map = {
        acc["HotelName"].split('(')[0].strip(): float(acc["Price"].replace("¥", ""))
        for acc in hotels if "Price" in acc
    }
    hotel_capacity_map = {
        acc["HotelName"].split('(')[0].strip(): int(acc["MaximumOccupancy"])
        for acc in hotels if "MaximumOccupancy" in acc
    }

    for day in predicted_entry["plan"][:-1]:  # 最后一天不计算住宿
        acc_name = parse_accommodation_info(day.get("accommodation", "-"))
        if acc_name == "-":
            continue

        matched_hotels = [name for name in hotel_price_map.keys() if acc_name in name]
        if matched_hotels:
            price = hotel_price_map[matched_hotels[0]]
            capacity = hotel_capacity_map.get(matched_hotels[0], 2)  # 默认每间房2人
            rooms = math.ceil(people / capacity)
            total_cost += price * rooms

# ---------------------- 3. 计算餐饮费用 ----------------------
    restaurants = ref_info.get(f"在{dest}的餐厅", [])
    # 默认餐饮费用为80元/人/餐
    DEFAULT_MEAL_COST = 80.0

    for day in predicted_entry["plan"]:
        # 午餐费用
        lunch = day.get("lunch", "-")
        if isinstance(lunch, list):
            lunch = lunch[0] if lunch else "-"
        if lunch != "-":
            # 查找餐厅价格，如果没有则使用默认值
            lunch_cost = next(
                (float(rest["Average Cost"]) 
                for rest in restaurants 
                if rest.get("Name") == lunch and rest.get("Average Cost") and isinstance(rest["Average Cost"], str)),
                DEFAULT_MEAL_COST
            )
            total_cost += lunch_cost * people

        # 晚餐费用
        dinner = day.get("dinner", "-")
        if isinstance(dinner, list):
            dinner = dinner[0] if dinner else "-"
        if dinner != "-":
            # 查找餐厅价格，如果没有则使用默认值
            dinner_cost = next(
                (float(rest["Average Cost"]) 
                for rest in restaurants 
                if rest.get("Name") == dinner and rest.get("Average Cost") and isinstance(rest["Average Cost"], str)),
                DEFAULT_MEAL_COST
            )
            total_cost += dinner_cost * people

    return round(total_cost, 2)
def is_valid_budget(test_entry: Dict[str, Any], predicted_entry: Dict[str, Any]) -> bool:
    """Validate if total cost is within budget"""
    try:
        budget = float(test_entry["budget"])
        total_cost = get_total_cost(test_entry, predicted_entry)
        return total_cost <= budget
    except (KeyError, ValueError, AttributeError):
        return False

def parse_local_constraint(local_constraint: str) -> Dict[str, str]:
    """Parse local constraint string into a dictionary"""
    constraints = {}
    if not isinstance(local_constraint, str) or not local_constraint:
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
            if not isinstance(predicted_entry, dict) or "travel_plan" not in predicted_entry:
                rewards[idx] = 0.0
                continue
                
            travel_plan = predicted_entry["travel_plan"]
            if not isinstance(travel_plan, list) or not travel_plan:  # Check if plan is empty
                rewards[idx] = 0.0
                continue
        except json.JSONDecodeError:
            rewards[idx] = 0.0  # Invalid JSON format
            continue
            
        # Create plan dictionary for validation functions
        plan_dict = {"plan": travel_plan}
        
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