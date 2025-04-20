import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Traffic Accident Dashboard", layout="wide")

# Inject styles
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
    .dashboard-header {
        background: linear-gradient(135deg, #2c3e50 0%, #4a6491 100%);
        padding: 30px; border-radius: 10px; color: white;
        text-align: center; margin-bottom: 30px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
    }
    .metric-card {
        background: linear-gradient(135deg, #E65C50 0%, #D14942 100%);
        color: white; border-radius: 10px; padding: 15px;
        text-align: center; margin: 10px 0; transition: transform 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('traffic_accidents_scattered_locations.csv')
        if df.empty:
            raise ValueError("Empty dataframe")
    except Exception as e:
        st.warning(f"Error loading data: {str(e)}. Using sample data.")
        np.random.seed(42)
        date_range = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        num_records = 2000
        cities = ['New York', 'Chicago', 'Los Angeles', 'Houston', 'Phoenix',
                  'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']
        city_coords = {
            'New York': (40.7128, -74.0060),
            'Chicago': (41.8781, -87.6298),
            'Los Angeles': (34.0522, -118.2437),
            'Houston': (29.7604, -95.3698),
            'Phoenix': (33.4484, -112.0740),
            'Philadelphia': (39.9526, -75.1652),
            'San Antonio': (29.4241, -98.4936),
            'San Diego': (32.7157, -117.1611),
            'Dallas': (32.7767, -96.7970),
            'San Jose': (37.3382, -121.8863)
        }

        city_choices = np.random.choice(cities, num_records)
        latitudes = [city_coords[city][0] + np.random.normal(0, 0.02) for city in city_choices]
        longitudes = [city_coords[city][1] + np.random.normal(0, 0.02) for city in city_choices]

        df = pd.DataFrame({
            'crash_date': np.random.choice(date_range, num_records),
            'damage': np.random.choice(['Minor', 'Moderate', 'Severe', 'Fatal'], num_records, p=[0.5, 0.3, 0.15, 0.05]),
            'roadway_surface_cond': np.random.choice(['Dry', 'Wet', 'Icy', 'Snow'], num_records, p=[0.6, 0.25, 0.1, 0.05]),
            'time_of_day': np.random.choice(['Morning (6am-12pm)', 'Afternoon (12pm-6pm)', 'Evening (6pm-12am)', 'Night (12am-6am)'], num_records),
            'city': city_choices,
            'latitude': latitudes,
            'longitude': longitudes
        })

    df['crash_date'] = pd.to_datetime(df['crash_date'])
    df['year'] = df['crash_date'].dt.year
    df['month'] = df['crash_date'].dt.month_name()
    df['month_num'] = df['crash_date'].dt.month
    df['day_of_week'] = df['crash_date'].dt.day_name()
    df['hour'] = df['crash_date'].dt.hour
    return df

df = load_data()

# Sidebar Filters
with st.sidebar:
    st.markdown("<h3><i class='fas fa-filter'></i> Dashboard Filters</h3>", unsafe_allow_html=True)

    with st.expander("🕰️ Time Period", expanded=True):
        if 'year' in df.columns:
            years = st.slider(
                "Select Year Range",
                min_value=int(df['year'].min()),
                max_value=int(df['year'].max()),
                value=(int(df['year'].min()), int(df['year'].max()))
            )
            df = df[df['year'].between(years[0], years[1])]

    with st.expander("🚗 Accident Details", expanded=True):
        if 'damage' in df.columns:
            damage_levels = st.multiselect(
                "Damage Severity",
                options=sorted(df['damage'].unique()),
                default=sorted(df['damage'].unique())
            )
            df = df[df['damage'].isin(damage_levels)]

        if 'roadway_surface_cond' in df.columns:
            road_conditions = st.multiselect(
                "Road Conditions",
                options=sorted(df['roadway_surface_cond'].unique()),
                default=sorted(df['roadway_surface_cond'].unique())
            )
            df = df[df['roadway_surface_cond'].isin(road_conditions)]

    with st.expander("🏙️ City", expanded=True):
        if 'city' in df.columns:
            city_list = sorted(df['city'].unique())
            selected_cities = st.multiselect("Select Cities", options=city_list, default=city_list)
            df = df[df['city'].isin(selected_cities)]


st.markdown("<h2 class='dashboard-header'>🚦 Traffic Accident Dashboard</h2>", unsafe_allow_html=True)

# Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""<div class="metric-card"><i class="fas fa-calendar-alt fa-2x"></i><h3>{df['year'].nunique()}</h3><p>Years of Data</p></div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-card"><i class="fas fa-car-crash fa-2x"></i><h3>{len(df):,}</h3><p>Total Accidents</p></div>""", unsafe_allow_html=True)
with col3:
    peak_month = df['month'].mode()[0]
    st.markdown(f"""<div class="metric-card"><i class="fas fa-calendar-check fa-2x"></i><h3>{peak_month}</h3><p>Peak Month</p></div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="metric-card"><i class="fas fa-road fa-2x"></i><h3>{df['roadway_surface_cond'].nunique()}</h3><p>Road Conditions</p></div>""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["📈 Trends Analysis", "🗺️ Geographic View", "🔍 Deep Dive"])

# ----- TAB 1: Trends -----
with tab1:
    st.markdown("### 📅 Monthly Accident Trends")
    chart_type = st.selectbox("Select Visualization Type", ["Area Chart", "Bar Chart", "Line Chart", "Violin Plot"])
    monthly_data = df.groupby(['year', 'month', 'month_num']).size().reset_index(name='count')
    monthly_data['month'] = pd.Categorical(monthly_data['month'], categories=[
        'January','February','March','April','May','June','July','August','September','October','November','December'
    ], ordered=True)
    monthly_data = monthly_data.sort_values(['year', 'month_num'])

    if chart_type == "Area Chart":
        fig = px.area(monthly_data, x='month', y='count', color='year', title="Monthly Accident Trends", color_discrete_sequence=px.colors.sequential.Reds)
    elif chart_type == "Bar Chart":
        fig = px.bar(monthly_data, x='month', y='count', color='year', barmode='group', title="Monthly Accident Trends", color_discrete_sequence=px.colors.sequential.Reds)
    elif chart_type == "Line Chart":
        fig = px.line(monthly_data, x='month', y='count', color='year', title="Monthly Accident Trends", color_discrete_sequence=px.colors.sequential.Reds)
    else:
        fig = px.violin(df, x='month', y='year', color='year', box=True, points="all", title="Accident Distribution by Month", color_discrete_sequence=px.colors.sequential.Reds)
    st.plotly_chart(fig, use_container_width=True)

# ----- TAB 2: Map -----
with tab2:
    st.markdown("### 🗺️ Accident Distribution by Severity within Each City")

    # Manually define approximate coordinates for cities (still needed for cities without exact coordinates)
    city_coords = {
        "New York": (40.7128, -74.0060),
        "Los Angeles": (34.0522, -118.2437),
        "Chicago": (41.8781, -87.6298),
        "Houston": (29.7604, -95.3698),
        "Phoenix": (33.4484, -112.0740),
        "Philadelphia": (39.9526, -75.1652),
        "San Antonio": (29.4241, -98.4936),
        "San Diego": (32.7157, -117.1611),
        "Dallas": (32.7767, -96.7970),
        "San Jose": (37.3382, -121.8863)
    }

    # Filter to cities in the mapping
    filtered_df = df[df['city'].isin(city_coords.keys())].copy()

    if filtered_df.empty:
        st.warning("No accident data for mapped cities based on the current filters.")
    else:
        # Allow user to select damage severity for the map (this filter will still apply)
        severity_options = sorted(filtered_df['damage'].unique())
        selected_severity_map = st.multiselect("Filter by Damage Severity for Map", severity_options, default=severity_options)
        map_data = filtered_df[filtered_df['damage'].isin(selected_severity_map)].copy() # Use .copy() here too

        if map_data.empty:
            st.info("No accidents with the selected severity in the chosen cities.")
        else:
            # Check if 'crash_latitude' and 'crash_longitude' columns exist
            if 'crash_latitude' in map_data.columns and 'crash_longitude' in map_data.columns:
                # Create map showing individual accidents colored by severity using actual coordinates
                fig = px.scatter_mapbox(
                    map_data,
                    lat="crash_latitude",  # Use the correct column name
                    lon="crash_longitude", # Use the correct column name
                    color="damage",
                    category_orders={"damage": severity_options},
                    color_discrete_sequence=px.colors.cyclical.IceFire,
                    size_max=15,
                    zoom=3,
                    height=600,
                    mapbox_style="open-street-map",
                    title="Accident Distribution by Severity within Each City"
                )
            else:
                # Fallback to city-level plotting if coordinates are missing
                city_summary = (
                    map_data.groupby(['city', 'damage'])
                    .size()
                    .reset_index(name='accident_count')
                )
                city_summary["lat"] = city_summary["city"].apply(lambda x: city_coords[x][0])
                city_summary["lon"] = city_summary["city"].apply(lambda x: city_coords[x][1])

                fig = px.scatter_mapbox(
                    city_summary,
                    lat="lat",
                    lon="lon",
                    size="accident_count",
                    hover_name="city",
                    color="damage",
                    category_orders={"damage": severity_options},
                    color_discrete_sequence=px.colors.cyclical.IceFire,
                    size_max=40,
                    zoom=3,
                    height=600,
                    mapbox_style="open-street-map",
                    title="Accident Distribution by Severity within Each City (City-Level)"
                )

            fig.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0})
            st.plotly_chart(fig, use_container_width=True)


# ----- TAB 3: Deep Dive -----
with tab3:
    st.markdown("### 🔍 Detailed Analysis")
    col1, col2 = st.columns(2)
    with col1:
        condition_data = df.groupby(['roadway_surface_cond', 'damage']).size().reset_index(name='count')
        fig = px.bar(condition_data, x='roadway_surface_cond', y='count', color='damage', barmode='group', title="Road Condition Impact", color_discrete_sequence=px.colors.sequential.Reds)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        severity_trend = df.groupby(['year', 'damage']).size().reset_index(name='count')
        fig = px.line(severity_trend, x='year', y='count', color='damage', title="Severity Over Time", color_discrete_sequence=px.colors.sequential.Reds)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 📊 Raw Data Explorer")
    with st.expander("View and Filter Data"):
        st.data_editor(df, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 14px; margin-top: 30px;">
    <p>Traffic Accident Analysis Dashboard • Created with Streamlit</p>
    <p><i class="fas fa-database"></i> Data Source: Sample / Kaggle</p>
</div>
""", unsafe_allow_html=True)
