import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


class UserDataGenerator:
    """生成模拟用户数据"""

    def __init__(self):
        self.user_ids = [f"user_{i:04d}" for i in range(1, 1001)]

    def generate_users(self, n_users=1000):
        """生成用户画像数据"""
        users = []
        for i in range(n_users):
            user = {
                "user_id": self.user_ids[i],
                "age": random.randint(18, 65),
                "gender": random.choice(["M", "F"]),
                "signup_date": (datetime.now() - timedelta(days=random.randint(1, 365))).strftime("%Y-%m-%d"),
                "total_spent": round(random.uniform(0, 2000), 2),
                "visit_count": random.randint(1, 50),
                "last_visit": (datetime.now() - timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%d"),
                "preferred_category": random.choice(["electronics", "fashion", "home", "beauty", "sports"]),
                "region": random.choice(["North", "South", "East", "West"])
            }
            users.append(user)

        # 保存数据
        with open('data/sample_users.json', 'w') as f:
            json.dump(users, f, indent=2)

        return pd.DataFrame(users)

    def generate_events(self, n_events=5000):
        """生成用户行为事件"""
        events = []
        event_types = ["page_view", "product_view", "add_to_cart", "purchase", "search"]

        for _ in range(n_events):
            event = {
                "user_id": random.choice(self.user_ids),
                "event_type": random.choice(event_types),
                "timestamp": (datetime.now() - timedelta(hours=random.randint(0, 720))).isoformat(),
                "product_category": random.choice(["electronics", "fashion", "home", "beauty", "sports"]),
                "value": round(random.uniform(0, 500), 2) if random.random() > 0.7 else 0
            }
            events.append(event)

        df_events = pd.DataFrame(events)
        df_events.to_csv('data/user_events.csv', index=False)
        return df_events