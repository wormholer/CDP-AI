import pandas as pd
from models.user_clustering import UserClusterAnalyzer


class SegmentManager:
    """AI驱动的用户分群管理"""

    def __init__(self, predictor):
        self.predictor = predictor
        self.segments = {}

    def create_ai_segments(self, users_df):
        """创建智能用户分群"""

        # 1. 预测用户价值
        features_df = self.predictor.prepare_features(users_df, pd.DataFrame())
        predictions = []

        for _, user in features_df.iterrows():
            pred = self.predictor.predict(pd.DataFrame([user]))
            predictions.append(pred)

        users_df['ai_segment'] = [p['segment'] for p in predictions]
        users_df['value_probability'] = [p['probability'] for p in predictions]

        # 2. 聚类分析
        cluster_analyzer = UserClusterAnalyzer()
        clusters = cluster_analyzer.cluster_users(users_df)
        users_df['behavior_cluster'] = clusters

        # 3. 定义综合分群
        def define_segment(row):
            if row['ai_segment'] == '高价值用户' and row['behavior_cluster'] == 0:
                return "核心忠实用户"
            elif row['ai_segment'] == '高价值用户':
                return "高价值潜力用户"
            elif row['value_probability'] > 0.3:
                return "成长中用户"
            else:
                return "普通用户"

        users_df['final_segment'] = users_df.apply(define_segment, axis=1)

        # 统计分群结果
        segment_stats = users_df['final_segment'].value_counts().to_dict()
        print("📊 AI用户分群完成:")
        for segment, count in segment_stats.items():
            print(f"   - {segment}: {count}人")

        self.segments = users_df
        return users_df

    def get_segment_recommendations(self, segment_name):
        """获取分群运营建议"""
        recommendations = {
            "核心忠实用户": "提供VIP专属优惠和提前访问权限，提升忠诚度",
            "高价值潜力用户": "推送个性化产品和限时优惠，促进转化",
            "成长中用户": "通过内容营销和教育材料培养兴趣",
            "普通用户": "发送通用促销信息和品牌内容，提高参与度"
        }
        return recommendations.get(segment_name, "暂无特定建议")