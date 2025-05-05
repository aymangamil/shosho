import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Merge the data
train=pd.read_csv(r"train_df.csv")
test=pd.read_csv(r"test_df.csv")
df = pd.concat([train, test], axis=0)

select=st.sidebar.selectbox('choose Option',['EDA'])

# Set up variables
st.image(r"analysis-concept-drawn-on-white-260nw-488594248.webp")
st.markdown("**By: Shahoda ❤️**")
target_column = 'satisfaction'  # Replace this with your actual target column

numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
df[target_column] = df[target_column].map({
    'satisfied': 1,
    'neutral or dissatisfied': 0
})
# Tabs for full analysis
tab1, tab2, tab3 = st.tabs(["📈 Numerical Analysis", "📁 Categorical Analysis", "📊 Correlation with Target"])

# =============== Tab 1: Numerical =============== #
with tab1:
    st.subheader("📊 Numerical Column Analysis")

    sub_tab1, sub_tab2 = st.tabs(["📊 Distribution & Stats", "🚨 Outliers Summary"])
    
    # -------- Sub-tab: Stats & Histogram -------- #
    with sub_tab1:
        for col in numerical_cols[1:]:
            st.markdown(f"### 🔢 Column: `{col}`")

            col_data = df[col].dropna()
            mean_val = round(col_data.mean(), 2)
            median_val = round(col_data.median(), 2)
            mode_val = round(col_data.mode()[0], 2)
            std_val = round(col_data.std(), 2)
            missing_val = df[col].isnull().sum()

            insight = ""
            if abs(mean_val - median_val) > std_val * 0.5:
                insight = f"⚠️ `{col}` is **skewed** (mean and median differ)."
            elif df[col].nunique() < 10:
                insight = f"ℹ️ `{col}` may be categorical (only {df[col].nunique()} unique values)."
            else:
                insight = f"✅ `{col}` seems fairly distributed."

            st.markdown(f"""
            - **Mean**: {mean_val}  
            - **Median**: {median_val}  
            - **Mode**: {mode_val}  
            - **Std Dev**: {std_val}  
            - **Missing Values**: {missing_val}  
            - **Insight**: {insight}
            """)

            fig = px.histogram(df, x=col, nbins=30, color_discrete_sequence=['#00b894'])
            fig.add_vline(x=mean_val, line_dash="dash", line_color="blue", annotation_text="Mean")
            fig.add_vline(x=median_val, line_dash="dot", line_color="orange", annotation_text="Median")
            fig.add_vline(x=mean_val + std_val, line_dash="dash", line_color="gray", annotation_text="+1 STD")
            fig.add_vline(x=mean_val - std_val, line_dash="dash", line_color="gray", annotation_text="-1 STD")
            st.plotly_chart(fig, use_container_width=True)

    # -------- Sub-tab: Outliers -------- #
    with sub_tab2:
        st.subheader("🚨 Outlier Analysis using IQR")
        outlier_data = []

        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower) | (df[col] > upper)][col]
            outlier_data.append({
                'Column': col,
                'Outliers Count': len(outliers),
                'Outliers %': round(len(outliers) / df[col].count() * 100, 2)
            })

        outlier_df = pd.DataFrame(outlier_data)
        st.dataframe(outlier_df.style.format({"Outliers %": "{:.2f}%"}))

        fig = px.bar(outlier_df, x='Column', y='Outliers Count', color='Outliers %', text='Outliers %',
                     color_continuous_scale='Teal', template='plotly_white')
        fig.update_layout(yaxis_title="Count", xaxis_title="Column")
        st.plotly_chart(fig, use_container_width=True)

# =============== Tab 2: Categorical =============== #
with tab2:
    st.subheader("📁 Categorical Column Analysis")

    for col in categorical_cols:
        st.markdown(f"### 📁 Column: `{col}`")

        value_counts = df[col].value_counts(dropna=False)
        top_3 = value_counts.head(3)
        total = value_counts.sum()
        percentages = (top_3 / total * 100).round(2)

        result_df = pd.DataFrame({
            'Category': top_3.index.astype(str),
            'Count': top_3.values,
            'Percentage': percentages.values
        })

        st.dataframe(result_df, use_container_width=True)

        fig = px.bar(result_df, x='Category', y='Count', text='Percentage', color='Category',
                     color_discrete_sequence=['#00b894', '#55efc4', '#81ecec'],
                     title=f"Top 3 Values in `{col}`")
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        fig.update_layout(showlegend=False, xaxis_title='Category', yaxis_title='Count',
                          plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

        top_cat = result_df.loc[0, 'Category']
        top_pct = result_df.loc[0, 'Percentage']
        unique_count = df[col].nunique()

        if top_pct >= 70:
            insight = f"🔎 `{col}` is highly dominated by `{top_cat}` ({top_pct}%)."
        elif top_pct >= 40:
            insight = f"🔍 Frequent value `{top_cat}` in `{col}` appears {top_pct}% — group rare values?"
        elif unique_count > 10:
            insight = f"📌 `{col}` has many unique values ({unique_count})."
        else:
            insight = f"🧩 `{col}` is fairly balanced."
        st.markdown(f"**Insight:** {insight}")

# =============== Tab 3: Correlation with Target =============== #
with tab3:
    st.subheader(f"📊 Top 10 Features Correlated with `{target_column}`")

    corr = df[numerical_cols.tolist() + [target_column]].corr()[target_column].drop(target_column)
    top_corr = corr.abs().sort_values(ascending=False).head(10)
    corr_df = pd.DataFrame({
        'Feature': top_corr.index,
        'Correlation': corr[top_corr.index].values
    })

    st.dataframe(corr_df.style.format({"Correlation": "{:.3f}"}), use_container_width=True)

    fig = px.bar(corr_df, x='Feature', y='Correlation', text='Correlation',
                 color='Correlation', color_continuous_scale='Viridis',
                 title="Top 10 Correlated Features")
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(showlegend=False, yaxis_title="Correlation with Target", plot_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True)