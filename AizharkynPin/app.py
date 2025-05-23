import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv("data (1).csv")
    df = df.dropna(subset=['MSRP'])
    return df

data = load_data()


@st.cache_resource
def train_model(df):
    df = df.dropna(subset=[
        'Make', 'Model', 'Year', 'Engine Fuel Type', 'Engine HP', 'Engine Cylinders',
        'Transmission Type', 'Driven_Wheels', 'MSRP'
    ])

    X = df[['Make', 'Model', 'Year', 'Engine Fuel Type', 'Engine HP',
            'Engine Cylinders', 'Transmission Type', 'Driven_Wheels']]
    y = df['MSRP']

    cat_features = ['Make', 'Model', 'Engine Fuel Type', 'Transmission Type', 'Driven_Wheels']
    num_features = ['Year', 'Engine HP', 'Engine Cylinders']

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])

    pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('model', RandomForestRegressor(n_estimators=200, random_state=42))
    ])

    pipeline.fit(X, y)
    return pipeline

model = train_model(data)


st.title("🚗 Car Price Predictor")
st.markdown("Predict the price of your dream car using AI")

tab1, tab2 = st.tabs(["Price Prediction", "Data Exploration"])

with tab1:
    st.subheader("Select Car Features")

    col1, col2 = st.columns(2)

    with col1:
        make = st.selectbox("Make", sorted(data['Make'].dropna().unique()))
        models = data[data['Make'] == make]['Model'].dropna().unique()
        model_name = st.selectbox("Model", sorted(models))

        years = sorted(data['Year'].dropna().unique(), reverse=True)


        default_year = 2020 if 2020 in years else years[0]

        year = st.select_slider("Year", options=years, value=default_year)

        fuel_type = st.selectbox("Fuel Type", sorted(data['Engine Fuel Type'].dropna().unique()))

    with col2:
        hp = st.slider("Engine HP", int(data['Engine HP'].min()), int(data['Engine HP'].max()), 200)
        cylinders = st.radio("Cylinders", sorted(data['Engine Cylinders'].dropna().unique()), horizontal=True)
        transmission = st.selectbox("Transmission", sorted(data['Transmission Type'].dropna().unique()))
        drive = st.selectbox("Driven Wheels", sorted(data['Driven_Wheels'].dropna().unique()))

    if st.button("🔮 Predict Price"):
        input_df = pd.DataFrame([{
            'Make': make,
            'Model': model_name,
            'Year': year,
            'Engine Fuel Type': fuel_type,
            'Engine HP': hp,
            'Engine Cylinders': cylinders,
            'Transmission Type': transmission,
            'Driven_Wheels': drive
        }])

        prediction = model.predict(input_df)[0]

        st.markdown(f"""
        <div style="background-color: #f0f8ff; padding: 20px; border-radius: 10px; border-left: 5px solid #2a9d8f;">
            <h3>Predicted Price:</h3>
            <h1 style="color: #2a9d8f;">${prediction:,.2f}</h1>
            <p>Based on selected car features.</p>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.subheader("Explore the Dataset")

    if st.checkbox("Show Raw Data"):
        st.dataframe(data)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.box(data, x='Make', y='MSRP', title="Price by Make")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(data, x='Engine HP', y='MSRP', color='Make', title="Price vs Horsepower")
        st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(data, x='Year', title="Car Count by Year")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("<center>© 2025 Car Price Predictor | Powered by Streamlit & Scikit-learn</center>", unsafe_allow_html=True)
