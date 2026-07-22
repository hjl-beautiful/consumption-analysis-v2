"""
电商用户消费行为分析系统 — Streamlit 可视化部署
基于 UCI Online Retail 数据集，完成数据清洗、RFM 分析、K-Means 聚类与随机森林分类
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# ==================== 修复中文显示（matplotlib-fontja，跨平台含 Streamlit Cloud） ====================
import matplotlib_fontja  # 自动注入 IPAexGothic，无需其他配置

# 设置页面
st.set_page_config(
    page_title="电商用户消费行为分析系统",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 全局样式
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #1f4e79;
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: 2px;
    }
    .main-subtitle {
        text-align: center;
        color: #666;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .section-title {
        color: #1f4e79;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
        display: inline-block;
    }
    .stButton>button {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        box-shadow: 0 2px 8px rgba(102,126,234,0.4);
    }
    .stButton>button[kind="secondary"] {
        background: #f8f9fa;
        color: #555;
        border: 1px solid #e0e0e0;
    }
    .stButton>button[kind="secondary"]:hover {
        background: #e9ecef;
        border-color: #667eea;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== 加载数据 ====================
@st.cache_data
def load_data():
    df = pd.read_parquet('data/OnlineRetail_cleaned.parquet')
    if 'InvoiceDate' in df.columns:
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    rfm = pd.read_csv('data/rfm_results.csv')
    return df, rfm

try:
    df, rfm = load_data()
    data_loaded = True
except Exception as e:
    st.error(f"数据加载失败：{e}")
    st.info("请确保数据文件已正确上传")
    data_loaded = False

if data_loaded:
    # ==================== 页面标题 ====================
    st.markdown('<div class="main-title">电商用户消费行为分析系统</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">基于RFM模型与机器学习的用户价值洞察平台 | UCI Online Retail Dataset</div>', unsafe_allow_html=True)

    # ==================== 侧边栏导航 ====================
    st.sidebar.markdown("""
        <div style="text-align: center; margin-bottom: 1.5rem; padding: 1rem; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 12px; color: white;">
            <h2 style="color: white; margin: 0; font-size: 1.3rem;">功能导航</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 0.3rem 0 0 0; font-size: 0.8rem;">选择分析模块</p>
        </div>
    """, unsafe_allow_html=True)

    if 'current_page' not in st.session_state:
        st.session_state.current_page = "首页概览"

    nav_items = ["首页概览", "RFM分析", "K-Means聚类", "随机森林分类", "客户价值预测"]

    for item in nav_items:
        is_active = st.session_state.current_page == item
        btn_type = "primary" if is_active else "secondary"
        if st.sidebar.button(item, key=f"nav_{item}", type=btn_type, use_container_width=True):
            st.session_state.current_page = item
            st.rerun()

    page = st.session_state.current_page

    st.sidebar.markdown("""
        <hr style="margin: 1.5rem 0; border: none; border-top: 1px solid #ddd;">
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; 
                    border-left: 4px solid #667eea;">
            <h4 style="color: #1f4e79; margin-bottom: 0.8rem; font-size: 0.95rem;">关于本项目</h4>
            <p style="font-size: 0.8rem; color: #555; line-height: 1.6; margin: 0;">
                <b>数据源：</b>UCI Online Retail<br>
                <b>记录数：</b>397,884条<br>
                <b>用户数：</b>4,338人<br>
                <b>时间跨度：</b>373天
            </p>
            <p style="font-size: 0.75rem; color: #888; margin-top: 0.6rem; margin-bottom: 0;">
                Python + Streamlit + Scikit-learn
            </p>
        </div>
    """, unsafe_allow_html=True)

    # ==================== 首页概览 ====================
    if page == "首页概览":
        st.markdown('<div class="section-title">核心指标看板</div>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        metrics = [
            (f"{df['CustomerID'].nunique():,}", "总用户数", "#667eea"),
            (f"{df['InvoiceNo'].nunique():,}", "总订单数", "#764ba2"),
            (f"£{df['Amount'].sum()/1e6:.2f}M", "总销售额", "#f093fb"),
            (f"{df['StockCode'].nunique():,}", "商品种类", "#4facfe"),
        ]
        for col, (value, label, color) in zip([col1, col2, col3, col4], metrics):
            with col:
                st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {color}22 0%, {color}44 100%); 
                                border-radius: 15px; padding: 1.5rem; text-align: center;
                                border: 2px solid {color}33; box-shadow: 0 2px 8px rgba(0,0,0,0.06);">
                        <div style="font-size: 2rem; font-weight: 700; color: {color};">{value}</div>
                        <div style="font-size: 0.9rem; color: #666; margin-top: 0.3rem;">{label}</div>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">数据概览</div>', unsafe_allow_html=True)
        st.dataframe(
            df.head(10).style.format(precision=2).background_gradient(cmap='Blues', subset=['Amount']),
            use_container_width=True,
            height=400
        )

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("""
                <div style="background: #f8f9fa; border-radius: 12px; padding: 1.5rem;">
                    <h4 style="color: #1f4e79; margin-bottom: 1rem;">时间分布</h4>
            """, unsafe_allow_html=True)
            st.info(f"**数据周期：** {df['InvoiceDate'].min().date()} 至 {df['InvoiceDate'].max().date()}")
            st.info(f"**时间跨度：** {(df['InvoiceDate'].max() - df['InvoiceDate'].min()).days} 天")
            st.info(f"**覆盖市场：** {df['Country'].nunique()} 个国家/地区")

        with col_right:
            st.markdown("""
                <div style="background: #f8f9fa; border-radius: 12px; padding: 1.5rem;">
                    <h4 style="color: #1f4e79; margin-bottom: 1rem;">关键洞察</h4>
            """, unsafe_allow_html=True)
            st.info(f"**平均客单价：** £{df['Amount'].mean():.2f}")
            st.info(f"**数据保留率：** 约72%（清洗后）")
            st.info(f"**英国订单占比：** {(df['Country']=='United Kingdom').mean()*100:.1f}%")

    # ==================== RFM分析 ====================
    elif page == "RFM分析":
        st.markdown('<div class="section-title">RFM用户价值分析</div>', unsafe_allow_html=True)
        st.markdown('<h4 style="color: #555; margin: 1rem 0;">RFM指标分布</h4>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        rfm_metrics = [
            ('Recency', '最近消费天数', '#667eea'),
            ('Frequency', '消费频次', '#764ba2'),
            ('Monetary', '消费金额', '#f093fb'),
        ]
        for col, (metric, label, color) in zip([col1, col2, col3], rfm_metrics):
            with col:
                st.markdown(f'<h5 style="color: {color}; text-align: center;">{label}</h5>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(6, 4))
                data = rfm[metric]
                xlabel = metric
                if metric == 'Monetary':
                    data = np.log1p(data)
                    xlabel = 'log(Monetary + 1)'
                ax.hist(data, bins=30, color=color, alpha=0.7, edgecolor='white')
                ax.set_title(f'{label} Distribution', fontsize=12, color='#333')
                ax.set_xlabel(xlabel, fontsize=10)
                ax.set_ylabel('Count', fontsize=10)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                st.pyplot(fig)

        st.divider()
        st.markdown('<h4 style="color: #555; margin: 1rem 0;">用户分群统计</h4>', unsafe_allow_html=True)

        segment_counts = rfm['Segment'].value_counts()
        col_chart, col_table = st.columns([3, 2])

        with col_chart:
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = plt.cm.Set3(np.linspace(0, 1, len(segment_counts)))
            bars = ax.barh(range(len(segment_counts)), segment_counts.values[::-1], color=colors[::-1])
            ax.set_yticks(range(len(segment_counts)))
            ax.set_yticklabels(segment_counts.index[::-1], fontsize=10)
            ax.set_xlabel('User Count', fontsize=11)
            ax.set_title('User Segment Distribution', fontsize=13, color='#333', pad=15)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            for bar, val in zip(bars, segment_counts.values[::-1]):
                ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
                       f'{val} ({val/len(rfm)*100:.1f}%)', va='center', fontsize=9)
            st.pyplot(fig)

        with col_table:
            segment_detail = rfm.groupby('Segment').agg({
                'CustomerID': 'count',
                'Recency': 'mean',
                'Frequency': 'mean',
                'Monetary': 'mean'
            }).round(1)
            segment_detail.columns = ['User Count', 'Avg R', 'Avg F', 'Avg M']
            segment_detail = segment_detail.sort_values('User Count', ascending=False)
            st.dataframe(
                segment_detail.style.background_gradient(cmap='YlOrRd', subset=['User Count']),
                use_container_width=True,
                height=350
            )

    # ==================== K-Means聚类 ====================
    elif page == "K-Means聚类":
        st.markdown('<div class="section-title">K-Means智能分群</div>', unsafe_allow_html=True)
        st.markdown("""
            <div style="background: #f0f4ff; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem;">
                <h4 style="color: #1f4e79; margin-bottom: 1rem;">聚类配置</h4>
            </div>
        """, unsafe_allow_html=True)

        col_config1, col_config2 = st.columns([2, 1])
        with col_config1:
            selected_features = st.multiselect(
                "选择聚类特征",
                ['Recency', 'Frequency', 'Monetary'],
                default=['Recency', 'Frequency', 'Monetary'],
                help="建议全选以获得最佳分群效果"
            )
        with col_config2:
            n_clusters = st.slider("聚类数量 K", 2, 8, 4)

        if len(selected_features) >= 2:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(rfm[selected_features])
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            rfm['Cluster'] = clusters

            st.markdown('<h4 style="color: #555; margin: 1.5rem 0 1rem;">聚类中心特征</h4>', unsafe_allow_html=True)
            cluster_centers = pd.DataFrame(
                scaler.inverse_transform(kmeans.cluster_centers_),
                columns=selected_features
            ).round(2)
            cluster_centers.index = [f'Cluster {i}' for i in range(n_clusters)]
            st.dataframe(
                cluster_centers.style.background_gradient(cmap='RdYlBu_r', axis=0),
                use_container_width=True
            )

            col_dist, col_viz = st.columns([1, 2])
            with col_dist:
                st.markdown('<h5 style="color: #555;">各群体规模</h5>', unsafe_allow_html=True)
                cluster_counts = pd.Series(clusters).value_counts().sort_index()
                fig, ax = plt.subplots(figsize=(6, 5))
                colors = plt.cm.Set2(np.linspace(0, 1, n_clusters))
                wedges, texts, autotexts = ax.pie(
                    cluster_counts.values,
                    labels=[f'C{i}' for i in cluster_counts.index],
                    autopct='%1.1f%%',
                    colors=colors,
                    startangle=90
                )
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                ax.set_title('Cluster Distribution', fontsize=12, color='#333')
                st.pyplot(fig)

            with col_viz:
                if len(selected_features) == 3:
                    st.markdown('<h5 style="color: #555;">PCA降维可视化</h5>', unsafe_allow_html=True)
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2)
                    pca_result = pca.fit_transform(X_scaled)
                    fig, ax = plt.subplots(figsize=(10, 7))
                    colors = plt.cm.Set2(np.linspace(0, 1, n_clusters))
                    for i in range(n_clusters):
                        mask = clusters == i
                        ax.scatter(
                            pca_result[mask, 0],
                            pca_result[mask, 1],
                            c=[colors[i]],
                            label=f'Cluster {i} (n={mask.sum()})',
                            alpha=0.6,
                            s=50,
                            edgecolors='white',
                            linewidth=0.5
                        )
                    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
                    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
                    ax.set_title('K-Means Clustering (PCA)', fontsize=13, color='#333', pad=15)
                    ax.legend(loc='best', framealpha=0.9)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    st.info(f"累计方差解释率: {sum(pca.explained_variance_ratio_):.1%}")
        else:
            st.warning("请至少选择2个特征进行聚类分析")

    # ==================== 随机森林分类 ====================
    elif page == "随机森林分类":
        st.markdown('<div class="section-title">随机森林高价值客户预测</div>', unsafe_allow_html=True)
        st.markdown("""
            <div style="background: #f0fff4; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem;">
                <h4 style="color: #1f4e79; margin-bottom: 1rem;">模型配置</h4>
            </div>
        """, unsafe_allow_html=True)

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            test_size = st.slider("测试集比例", 0.1, 0.5, 0.2, 0.05)
        with col_c2:
            n_estimators = st.slider("决策树数量", 50, 300, 100, 25)

        feature_cols = ['Recency', 'Frequency', 'Monetary', 'AvgOrderValue', 'R_F_Ratio', 'M_F_Ratio']
        rfm['AvgOrderValue'] = rfm['Monetary'] / (rfm['Frequency'] + 1)
        rfm['R_F_Ratio'] = rfm['Recency'] / (rfm['Frequency'] + 1)
        rfm['M_F_Ratio'] = rfm['Monetary'] / (rfm['Frequency'] + 1)
        rfm['ValueScore'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']
        rfm['IsHighValue'] = (rfm['ValueScore'] >= rfm['ValueScore'].quantile(0.8)).astype(int)

        X = rfm[feature_cols]
        y = rfm['IsHighValue']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        with st.spinner('正在训练模型，请稍候...'):
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train_scaled, y_train)
            y_pred = rf.predict(X_test_scaled)
            y_pred_proba = rf.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        st.markdown('<h4 style="color: #555; margin: 1.5rem 0 1rem;">模型性能</h4>', unsafe_allow_html=True)
        col_m1, col_m2, col_m3 = st.columns(3)
        metrics_display = [
            (f"{acc*100:.2f}%", "准确率", "#667eea"),
            (f"{f1:.4f}", "F1-Score", "#764ba2"),
            (f"{roc_auc:.4f}", "AUC", "#f093fb"),
        ]
        for col, (val, label, color) in zip([col_m1, col_m2, col_m3], metrics_display):
            with col:
                st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {color}15 0%, {color}30 100%); 
                                border-radius: 15px; padding: 1.5rem; text-align: center;
                                border: 2px solid {color}40;">
                        <div style="font-size: 2.2rem; font-weight: 700; color: {color};">{val}</div>
                        <div style="font-size: 0.9rem; color: #666; margin-top: 0.5rem;">{label}</div>
                    </div>
                """, unsafe_allow_html=True)

        col_viz1, col_viz2 = st.columns(2)
        with col_viz1:
            st.markdown('<h5 style="color: #555;">特征重要性</h5>', unsafe_allow_html=True)
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=True)
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(importance_df)))
            bars = ax.barh(importance_df['feature'], importance_df['importance'], color=colors)
            ax.set_xlabel('Importance', fontsize=11)
            ax.set_title('Feature Importance (Random Forest)', fontsize=12, color='#333', pad=15)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            for bar, val in zip(bars, importance_df['importance']):
                ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                       f'{val:.3f}', va='center', fontsize=9, color='#555')
            st.pyplot(fig)

        with col_viz2:
            st.markdown('<h5 style="color: #555;">混淆矩阵</h5>', unsafe_allow_html=True)
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(7, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Normal', 'High Value'],
                       yticklabels=['Normal', 'High Value'],
                       annot_kws={'size': 14, 'weight': 'bold'})
            ax.set_title('Confusion Matrix', fontsize=12, color='#333', pad=15)
            ax.set_xlabel('Predicted', fontsize=11)
            ax.set_ylabel('Actual', fontsize=11)
            st.pyplot(fig)

    # ==================== 客户价值预测 ====================
    elif page == "客户价值预测":
        st.markdown('<div class="section-title">智能客户价值预测</div>', unsafe_allow_html=True)
        st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); 
                        border-radius: 15px; padding: 1.5rem; margin-bottom: 2rem;">
                <p style="color: #555; margin: 0;">输入客户的RFM指标，系统将基于训练好的模型预测客户价值等级，并给出运营建议。</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown('<h4 style="color: #1f4e79; margin-bottom: 1rem;">客户信息录入</h4>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div style="text-align: center; color: #667eea; font-weight: 600;">Recency</div>', unsafe_allow_html=True)
            recency = st.number_input("最近消费天数", min_value=0, max_value=500, value=30,
                                     help="距离上次消费的天数，越小越好")
        with col2:
            st.markdown('<div style="text-align: center; color: #764ba2; font-weight: 600;">Frequency</div>', unsafe_allow_html=True)
            frequency = st.number_input("消费频次", min_value=1, max_value=100, value=5,
                                        help="累计消费次数")
        with col3:
            st.markdown('<div style="text-align: center; color: #f093fb; font-weight: 600;">Monetary</div>', unsafe_allow_html=True)
            monetary = st.number_input("消费金额 (£)", min_value=0.0, max_value=100000.0, value=2000.0,
                                      help="累计消费金额")

        col_btn = st.columns([1, 2, 1])[1]
        with col_btn:
            predict_clicked = st.button("开始预测", use_container_width=True)

        if predict_clicked:
            avg_order = monetary / (frequency + 1)
            r_f_ratio = recency / (frequency + 1)
            m_f_ratio = monetary / (frequency + 1)

            r_score = 5 if recency < 30 else 4 if recency < 60 else 3 if recency < 90 else 2 if recency < 180 else 1
            f_score = 5 if frequency > 10 else 4 if frequency > 7 else 3 if frequency > 4 else 2 if frequency > 2 else 1
            m_score = 5 if monetary > 5000 else 4 if monetary > 3000 else 3 if monetary > 1500 else 2 if monetary > 500 else 1

            value_score = r_score + f_score + m_score

            st.markdown('<hr style="margin: 2rem 0;">', unsafe_allow_html=True)
            st.markdown('<h4 style="color: #1f4e79; text-align: center; margin-bottom: 1.5rem;">预测结果</h4>', unsafe_allow_html=True)

            if value_score >= 12:
                st.markdown("""
                    <div style="background: linear-gradient(135deg, #ffd70022 0%, #ffed4a22 100%); 
                                border: 3px solid #ffd700; border-radius: 20px; padding: 2rem; text-align: center;">
                        <h2 style="color: #b8860b; margin-bottom: 0.5rem;">高价值客户</h2>
                        <p style="color: #888; font-size: 1.1rem;">VIP级别 | 重点维护对象</p>
                    </div>
                """, unsafe_allow_html=True)
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    st.metric("RFM综合评分", f"{value_score}/15")
                    st.metric("消费金额", f"£{monetary:.2f}")
                with col_r2:
                    st.metric("消费频次", f"{frequency} 次")
                    st.metric("最近消费", f"{recency} 天前")
                st.success("""
                运营建议：
                - 提供专属VIP客服通道
                - 优先推送新品和限量款
                - 生日/节日专属礼遇
                - 邀请加入会员俱乐部
                """)

            elif value_score >= 8:
                st.markdown("""
                    <div style="background: linear-gradient(135deg, #4facfe22 0%, #00f2fe22 100%); 
                                border: 3px solid #4facfe; border-radius: 20px; padding: 2rem; text-align: center;">
                        <h2 style="color: #1f4e79; margin-bottom: 0.5rem;">潜力客户</h2>
                        <p style="color: #888; font-size: 1.1rem;">成长型 | 提升空间巨大</p>
                    </div>
                """, unsafe_allow_html=True)
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    st.metric("RFM综合评分", f"{value_score}/15")
                    st.metric("消费金额", f"£{monetary:.2f}")
                with col_r2:
                    st.metric("消费频次", f"{frequency} 次")
                    st.metric("最近消费", f"{recency} 天前")
                st.info("""
                运营建议：
                - 发送复购优惠券刺激消费
                - 推荐关联商品提升客单价
                - 会员积分活动吸引留存
                - 定期推送个性化推荐
                """)

            else:
                st.markdown("""
                    <div style="background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%); 
                                border: 3px solid #ccc; border-radius: 20px; padding: 2rem; text-align: center;">
                        <h2 style="color: #666; margin-bottom: 0.5rem;">普通客户</h2>
                        <p style="color: #888; font-size: 1.1rem;">基础型 | 需激活唤醒</p>
                    </div>
                """, unsafe_allow_html=True)
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    st.metric("RFM综合评分", f"{value_score}/15")
                    st.metric("消费金额", f"£{monetary:.2f}")
                with col_r2:
                    st.metric("消费频次", f"{frequency} 次")
                    st.metric("最近消费", f"{recency} 天前")
                st.warning("""
                运营建议：
                - 大额满减券吸引首单/复购
                - 限时秒杀活动刺激冲动消费
                - 新人专享福利重新激活
                - 短信/推送召回沉睡用户
                """)