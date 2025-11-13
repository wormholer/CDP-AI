#!/usr/bin/env python3
"""
CDP & AI 集成演示 - 主启动文件
"""

import warnings

warnings.filterwarnings('ignore')

from data.generate_data import UserDataGenerator
from models.train_model import UserBehaviorPredictor
from cdp_core.segment_manager import SegmentManager
from dashboard.app import start_dashboard


def main():
    """主演示流程"""
    print("🚀 启动 CDP & AI 智能用户预测平台...")

    # 1. 生成模拟数据
    print("📊 生成模拟用户数据...")
    generator = UserDataGenerator()
    user_data = generator.generate_users(1000)
    events_data = generator.generate_events(5000)

    # 2. 训练AI模型
    print("🤖 训练用户行为预测模型...")
    predictor = UserBehaviorPredictor()
    model_performance = predictor.train(user_data, events_data)

    print(f"✅ 模型训练完成 - 准确率: {model_performance['accuracy']:.3f}")

    # 3. 创建智能用户分群
    print("🎯 创建AI驱动的用户分群...")
    segment_manager = SegmentManager(predictor)
    segments = segment_manager.create_ai_segments(user_data)

    # 4. 启动可视化仪表板
    print("📈 启动数据可视化仪表板...")
    print("👉 请在浏览器中访问: http://localhost:8050")
    start_dashboard(user_data, segments, model_performance)


if __name__ == "__main__":
    main()