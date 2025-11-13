import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib


class UserBehaviorPredictor:
    """用户行为预测AI模型"""

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_columns = []

    def prepare_features(self, users_df, events_df):
        """准备模型特征"""
        # 用户基本特征
        users_df['is_active'] = (pd.to_datetime(users_df['last_visit']) >
                                 (pd.Timestamp.now() - pd.Timedelta(days=7))).astype(int)

        # 行为聚合特征
        user_behavior = events_df.groupby('user_id').agg({
            'event_type': 'count',
            'value': 'sum',
            'timestamp': lambda x: (pd.Timestamp.now() - pd.to_datetime(x.max())).days
        }).rename(columns={
            'event_type': 'total_events',
            'value': 'total_value',
            'timestamp': 'days_since_last_event'
        })

        # 合并特征
        features_df = users_df.merge(user_behavior, on='user_id', how='left')
        features_df = features_df.fillna(0)

        # 特征工程
        features_df['avg_order_value'] = np.where(
            features_df['total_events'] > 0,
            features_df['total_spent'] / features_df['visit_count'],
            0
        )

        # 编码分类变量
        features_df = pd.get_dummies(features_df,
                                     columns=['gender', 'preferred_category', 'region'])

        # 选择特征列
        exclude_cols = ['user_id', 'signup_date', 'last_visit']
        self.feature_columns = [col for col in features_df.columns
                                if col not in exclude_cols and not col.startswith('target_')]

        return features_df

    def create_target_variable(self, features_df):
        """创建预测目标 - 高价值用户"""
        # 基于消费金额和活跃度定义高价值用户
        spend_quantile = features_df['total_spent'].quantile(0.7)
        visit_quantile = features_df['visit_count'].quantile(0.7)

        features_df['target_high_value'] = (
                (features_df['total_spent'] > spend_quantile) &
                (features_df['visit_count'] > visit_quantile)
        ).astype(int)

        return features_df

    def train(self, users_df, events_df):
        """训练预测模型"""
        print("🛠️ 准备训练数据...")
        features_df = self.prepare_features(users_df, events_df)
        features_df = self.create_target_variable(features_df)

        X = features_df[self.feature_columns]
        y = features_df['target_high_value']

        # 分割数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print("🎯 训练随机森林模型...")
        self.model.fit(X_train, y_train)

        # 模型评估
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # 保存模型
        joblib.dump(self.model, 'models/user_value_model.pkl')
        print(f"✅ 模型保存完成 - 测试集准确率: {accuracy:.3f}")

        return {
            "accuracy": accuracy,
            "feature_importance": dict(zip(self.feature_columns,
                                           self.model.feature_importances_))
        }

    def predict(self, user_features):
        """预测单个用户价值"""
        prediction = self.model.predict(user_features[self.feature_columns])
        probability = self.model.predict_proba(user_features[self.feature_columns])

        return {
            "is_high_value": prediction[0],
            "probability": probability[0][1],
            "segment": "高价值用户" if prediction[0] else "普通用户"
        }