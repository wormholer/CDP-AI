import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def start_dashboard(users_df, segments_df, model_performance):
    """启动可视化仪表板"""

    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1("🎯 CDP & AI 智能用户分析平台",
                style={'textAlign': 'center', 'color': '#2C3E50'}),

        html.Div([
            # 模型性能卡片
            html.Div([
                html.H3("AI模型准确率"),
                html.H2(f"{model_performance['accuracy']:.1%}",
                        style={'color': '#27AE60'})
            ], className='card'),

            # 用户分群概览
            html.Div([
                html.H3("用户分群分布"),
                html.H2(f"{len(segments_df)}",
                        style={'color': '#2980B9'})
            ], className='card'),
        ], className='row'),

        # 分群分布图
        dcc.Graph(id='segment-pie'),

        # 特征重要性图
        dcc.Graph(id='feature-importance'),

        # 数据表格
        html.H3("用户分群详情"),
        html.Div(id='segment-table')
    ], style={'padding': '20px'})

    @app.callback(
        Output('segment-pie', 'figure'),
        Input('segment-table', 'children')
    )
    def update_pie_chart(_):
        segment_counts = segments_df['final_segment'].value_counts()
        fig = px.pie(values=segment_counts.values,
                     names=segment_counts.index,
                     title="AI用户分群分布")
        return fig

    @app.callback(
        Output('feature-importance', 'figure'),
        Input('feature-importance', 'id')
    )
    def update_feature_importance(_):
        # 获取特征重要性
        importance_data = model_performance['feature_importance']
        top_features = dict(sorted(importance_data.items(),
                                   key=lambda x: x[1], reverse=True)[:10])

        fig = go.Figure(data=[
            go.Bar(x=list(top_features.values()),
                   y=list(top_features.keys()),
                   orientation='h')
        ])
        fig.update_layout(title="AI模型特征重要性 Top 10")
        return fig

    app.run_server(debug=True, port=8050)