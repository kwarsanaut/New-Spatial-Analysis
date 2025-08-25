import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium import plugins
from streamlit_folium import st_folium
import os

# Page configuration
st.set_page_config(
    page_title="Restaurant Location ML System",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .location-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .stDataFrame {
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_models():
    """Load pre-trained models and data"""
    models_path = "models"
    
    try:
        # Load scalers
        with open(f"{models_path}/proper_scalers.pkl", 'rb') as f:
            scalers = pickle.load(f)
        
        # Load encoders
        with open(f"{models_path}/proper_encoders.pkl", 'rb') as f:
            encoders = pickle.load(f)
        
        # Load branch performance data
        branch_performance = pd.read_csv(f"{models_path}/proper_branch_performance.csv")
        
        # Load models
        models = {}
        for category in scalers.keys():
            filename = f"{models_path}/proper_model_{category.replace(' ', '_').replace('/', '_')}.pkl"
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    models[category] = pickle.load(f)
        
        return models, scalers, encoders, branch_performance
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

def create_location_map(predictions, category):
    """Create interactive map with location rankings"""
    if not predictions:
        return None
    
    # Create base map centered on Jakarta Selatan
    center_lat = -6.25
    center_lng = 106.8
    
    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Add markers for top locations
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'gray', 'black', 'lightred']
    
    for i, location in enumerate(predictions[:10]):
        color = colors[i % len(colors)]
        
        # Create popup text
        popup_text = f"""
        <b>Rank {i+1}: {location['district']}</b><br>
        Category: {category}<br>
        Coordinates: ({location['latitude']:.4f}, {location['longitude']:.4f})<br>
        <b>Est. Monthly Revenue: Rp {location['predicted_monthly_revenue']:,.0f}</b><br>
        <b>Opportunity Score: {location['opportunity_score']:,.0f}</b><br>
        Competitors: {location['existing_competitors']}<br>
        Market Density: {location['market_density']}
        """
        
        folium.Marker(
            [location['latitude'], location['longitude']],
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=f"Rank {i+1}: {location['district']}",
            icon=folium.Icon(color=color, icon='cutlery', prefix='fa')
        ).add_to(m)
    
    return m

def create_revenue_chart(predictions, category):
    """Create revenue comparison chart"""
    if not predictions:
        return None
    
    df_chart = pd.DataFrame(predictions[:10])
    df_chart['rank'] = range(1, len(df_chart) + 1)
    
    fig = go.Figure()
    
    # Monthly revenue bars
    fig.add_trace(go.Bar(
        x=df_chart['rank'],
        y=df_chart['predicted_monthly_revenue'],
        name='Monthly Revenue',
        text=[f"Rp {val:,.0f}" for val in df_chart['predicted_monthly_revenue']],
        textposition='auto',
        marker_color='lightblue'
    ))
    
    # Opportunity score line
    fig.add_trace(go.Scatter(
        x=df_chart['rank'],
        y=df_chart['opportunity_score'],
        mode='lines+markers',
        name='Opportunity Score',
        yaxis='y2',
        line=dict(color='red', width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title=f'Top 10 Revenue Predictions for {category}',
        xaxis_title="Location Rank",
        yaxis_title="Predicted Monthly Revenue (Rp)",
        yaxis2=dict(
            title="Opportunity Score",
            overlaying='y',
            side='right'
        ),
        height=500,
        showlegend=True
    )
    
    return fig

def create_competition_analysis(predictions, category):
    """Create competition vs revenue analysis"""
    if not predictions:
        return None
    
    df_chart = pd.DataFrame(predictions[:10])
    
    fig = px.scatter(
        df_chart, 
        x='existing_competitors', 
        y='predicted_monthly_revenue',
        size='market_density',
        hover_data=['district', 'opportunity_score'],
        title=f'{category} - Competition vs Revenue Analysis',
        labels={
            'existing_competitors': 'Number of Competitors',
            'predicted_monthly_revenue': 'Predicted Monthly Revenue (Rp)'
        }
    )
    
    fig.update_layout(height=400)
    return fig

def predict_locations(models, scalers, encoders, branch_performance, category, top_k=10):
    """Predict optimal locations for a category"""
    
    if category not in models:
        return None
    
    model = models[category]
    scaler = scalers[category]
    
    # Get successful branches for this category as benchmarks
    category_branches = branch_performance[branch_performance['category'] == category]
    
    if len(category_branches) == 0:
        return None
    
    # Define successful branches (top 40% performers)
    success_threshold = category_branches['monthly_revenue'].quantile(0.6)
    successful_branches = category_branches[category_branches['monthly_revenue'] >= success_threshold]
    
    # Calculate benchmarks
    benchmark_stats = {
        'avg_transaction_value': successful_branches['avg_transaction_value'].median(),
        'transactions_per_day': successful_branches['transactions_per_day'].median(),
        'revenue_per_day': successful_branches['revenue_per_day'].median(),
        'operational_intensity': successful_branches['operational_intensity'].median(),
        'transaction_consistency': successful_branches['transaction_consistency'].median(),
        'avg_items_per_transaction': successful_branches['avg_items_per_transaction'].median(),
        'peak_month': successful_branches['peak_month'].mode().iloc[0] if len(successful_branches) > 0 else 6,
        'peak_day_of_week': successful_branches['peak_day_of_week'].mode().iloc[0] if len(successful_branches) > 0 else 5
    }
    
    # Get unique potential locations (fix duplicate issue)
    potential_locations = branch_performance.groupby(['district']).agg({
        'latitude': 'mean',
        'longitude': 'mean', 
        'district_encoded': 'first',
        'district_avg_revenue': 'first',
        'district_avg_transaction_value': 'first',
        'category_median_revenue': 'first',
        'category_median_transaction_value': 'first'
    }).reset_index()
    
    predictions = []
    
    for idx, location in potential_locations.iterrows():
        # Calculate local competition for this specific category
        local_branches = branch_performance[
            (branch_performance['district'] == location['district']) &
            (branch_performance['category'] == category)
        ]
        
        competitor_count = len(local_branches)
        nearby_avg_revenue = local_branches['monthly_revenue'].mean() if competitor_count > 0 else 0
        nearby_avg_transaction_value = local_branches['avg_transaction_value'].mean() if competitor_count > 0 else location['district_avg_transaction_value']
        
        # Market density (all categories in district)
        all_local_branches = branch_performance[
            branch_performance['district'] == location['district']
        ]
        market_density = len(all_local_branches)
        
        # Performance ratios
        revenue_vs_district = 1.0
        revenue_vs_category = 1.0
        transaction_value_vs_category = benchmark_stats['avg_transaction_value'] / location['category_median_transaction_value']
        
        # Create feature vector matching training features
        features = [
            location['latitude'],
            location['longitude'],
            benchmark_stats['avg_transaction_value'],
            benchmark_stats['transactions_per_day'],
            benchmark_stats['revenue_per_day'],
            benchmark_stats['operational_intensity'],
            benchmark_stats['transaction_consistency'],
            benchmark_stats['avg_items_per_transaction'],
            benchmark_stats['peak_month'],
            benchmark_stats['peak_day_of_week'],
            competitor_count,
            nearby_avg_revenue,
            nearby_avg_transaction_value,
            market_density,
            location['district_encoded'],
            location['district_avg_revenue'],
            location['district_avg_transaction_value'],
            revenue_vs_district,
            revenue_vs_category,
            transaction_value_vs_category
        ]
        
        # Scale and predict
        features_array = np.array([features])
        features_scaled = scaler.transform(features_array)
        predicted_monthly_revenue = model.predict(features_scaled)[0]
        predicted_monthly_revenue = max(0, predicted_monthly_revenue)
        
        # Calculate opportunity score considering competition
        market_saturation_penalty = min(competitor_count * 0.1, 0.5)
        opportunity_score = predicted_monthly_revenue * (1 - market_saturation_penalty)
        
        predictions.append({
            'latitude': location['latitude'],
            'longitude': location['longitude'],
            'district': location['district'],
            'predicted_monthly_revenue': predicted_monthly_revenue,
            'opportunity_score': opportunity_score,
            'existing_competitors': competitor_count,
            'market_density': market_density,
            'predicted_annual_revenue': predicted_monthly_revenue * 12,
            'revenue_per_competitor': predicted_monthly_revenue / max(competitor_count + 1, 1)
        })
    
    # Sort by opportunity score
    predictions.sort(key=lambda x: x['opportunity_score'], reverse=True)
    return predictions[:top_k]

def main():
    # Header
    st.markdown('<h1 class="main-header">Restaurant Location ML System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Optimal Restaurant Location Prediction for Jakarta Selatan</p>', unsafe_allow_html=True)
    
    # Load models
    with st.spinner("Loading ML models..."):
        models, scalers, encoders, branch_performance = load_models()
    
    if models is None:
        st.error("Could not load trained models. Please ensure models are trained and saved.")
        st.info("Run the training script first to generate models.")
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Location Prediction", "Model Performance", "Market Analysis", "Data Overview"]
    )
    
    if page == "Location Prediction":
        st.header("Location Prediction Analysis")
        
        # Category selection
        available_categories = list(models.keys())
        selected_category = st.selectbox(
            "Select Restaurant Category",
            available_categories,
            help="Choose the restaurant category for location analysis"
        )
        
        # Parameters
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.subheader("Analysis Parameters")
            top_k = st.slider("Number of locations", 5, 10, 8)
            
            if st.button("Generate Predictions", type="primary"):
                with st.spinner(f"Analyzing optimal locations for {selected_category}..."):
                    predictions = predict_locations(
                        models, scalers, encoders, branch_performance, 
                        selected_category, top_k
                    )
                    
                    if predictions:
                        st.session_state.predictions = predictions
                        st.session_state.selected_category = selected_category
                        st.success(f"Analysis complete! Found {len(predictions)} optimal locations.")
                    else:
                        st.error("Failed to generate predictions.")
        
        with col1:
            if hasattr(st.session_state, 'predictions') and st.session_state.predictions:
                predictions = st.session_state.predictions
                category = st.session_state.selected_category
                
                st.subheader(f"Top Locations for {category}")
                
                # Display predictions table
                df_display = pd.DataFrame(predictions)
                df_display['Rank'] = range(1, len(df_display) + 1)
                df_display['Monthly Revenue'] = df_display['predicted_monthly_revenue'].apply(lambda x: f"Rp {x:,.0f}")
                df_display['Annual Revenue'] = df_display['predicted_annual_revenue'].apply(lambda x: f"Rp {x:,.0f}")
                df_display['Opportunity Score'] = df_display['opportunity_score'].apply(lambda x: f"{x:,.0f}")
                
                display_cols = ['Rank', 'district', 'Monthly Revenue', 'Annual Revenue', 
                               'existing_competitors', 'market_density', 'Opportunity Score']
                column_names = ['Rank', 'District', 'Monthly Revenue', 'Annual Revenue', 
                               'Competitors', 'Market Density', 'Opportunity Score']
                
                df_show = df_display[display_cols].copy()
                df_show.columns = column_names
                
                st.dataframe(df_show, use_container_width=True, hide_index=True)
        
        # Visualizations
        if hasattr(st.session_state, 'predictions') and st.session_state.predictions:
            predictions = st.session_state.predictions
            category = st.session_state.selected_category
            
            # Revenue chart
            st.subheader("Revenue Analysis")
            fig = create_revenue_chart(predictions, category)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Map and competition analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Location Map")
                location_map = create_location_map(predictions, category)
                if location_map:
                    st_folium(location_map, width=400, height=400)
            
            with col2:
                st.subheader("Competition Analysis")
                comp_fig = create_competition_analysis(predictions, category)
                if comp_fig:
                    st.plotly_chart(comp_fig, use_container_width=True)
    
    elif page == "Model Performance":
        st.header("Model Performance Dashboard")
        
        # Show available categories and their status
        st.subheader("Available Models")
        
        model_status = []
        for category in models.keys():
            model_status.append({
                'Category': category,
                'Status': 'Trained ✅',
                'Branches': len(branch_performance[branch_performance['category'] == category])
            })
        
        st.dataframe(pd.DataFrame(model_status), use_container_width=True, hide_index=True)
        
        st.info("Model performance metrics (R², MAE, RMSE) are displayed during training.")
    
    elif page == "Market Analysis":
        st.header("Market Analysis")
        
        # District analysis
        st.subheader("District Performance")
        
        district_stats = branch_performance.groupby('district').agg({
            'monthly_revenue': ['count', 'mean', 'median'],
            'avg_transaction_value': 'mean',
            'transactions_per_day': 'mean',
            'competitor_count': 'mean'
        }).round(0)
        
        district_stats.columns = ['Branch Count', 'Avg Revenue', 'Median Revenue', 'Avg Transaction Value', 'Avg Transactions/Day', 'Avg Competitors']
        district_stats = district_stats.reset_index()
        
        st.dataframe(district_stats, use_container_width=True, hide_index=True)
        
        # Category analysis
        st.subheader("Category Performance")
        
        category_stats = branch_performance.groupby('category').agg({
            'monthly_revenue': ['count', 'mean', 'median'],
            'avg_transaction_value': 'mean',
            'competitor_count': 'mean'
        }).round(0)
        
        category_stats.columns = ['Branch Count', 'Avg Revenue', 'Median Revenue', 'Avg Transaction Value', 'Avg Competitors']
        category_stats = category_stats.sort_values('Avg Revenue', ascending=False).reset_index()
        
        st.dataframe(category_stats, use_container_width=True, hide_index=True)
    
    elif page == "Data Overview":
        st.header("Data Overview")
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Branches", len(branch_performance['branch_id'].unique()))
        
        with col2:
            st.metric("Categories", len(branch_performance['category'].unique()))
        
        with col3:
            st.metric("Districts", len(branch_performance['district'].unique()))
        
        with col4:
            avg_revenue = branch_performance['monthly_revenue'].mean()
            st.metric("Avg Monthly Revenue", f"Rp {avg_revenue:,.0f}")
        
        # Revenue distribution
        st.subheader("Revenue Distribution")
        
        fig = px.histogram(
            branch_performance, 
            x='monthly_revenue', 
            nbins=50,
            title="Distribution of Monthly Revenue Across All Branches"
        )
        fig.update_xaxis(title="Monthly Revenue (Rp)")
        fig.update_yaxis(title="Number of Branches")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Top performing branches
        st.subheader("Top Performing Branches by Category")
        
        top_branches = branch_performance.nlargest(20, 'monthly_revenue')[
            ['branch_id', 'category', 'district', 'monthly_revenue', 'avg_transaction_value', 'transactions_per_day']
        ].reset_index(drop=True)
        
        top_branches['monthly_revenue'] = top_branches['monthly_revenue'].apply(lambda x: f"Rp {x:,.0f}")
        top_branches['avg_transaction_value'] = top_branches['avg_transaction_value'].apply(lambda x: f"Rp {x:,.0f}")
        top_branches['transactions_per_day'] = top_branches['transactions_per_day'].apply(lambda x: f"{x:.1f}")
        
        st.dataframe(top_branches, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()