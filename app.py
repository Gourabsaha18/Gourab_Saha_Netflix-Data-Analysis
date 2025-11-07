import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import plotly.graph_objects as go
import plotly.express as px

import streamlit as st

st.set_page_config(
    page_title="Netflix Data Analysis",
    page_icon="🎬",
    layout="wide",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)
# -------------------------------
# Load dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("netflix_titles.csv")
    return df

df = load_data()

# -------------------------------
# Sidebar Menu
# -------------------------------
st.sidebar.title("Netflix Data Analysis")
menu = [
    "Dataset Preview",
    "Missing Data Heatmap",
    "Data Cleaning",
    "Movies vs TV Shows",
    "Titles Added Per Year",
    "Top 10 Genres",
    "Top 10 Countries",
    "Ratings Distribution",
    "Movie Duration Distribution",
    "TV Show Season Counts",
    "Top 10 Actors",
    "Top 10 Directors",
    "User Choice Analysis",
    "Correlation Heatmap",
    "Global Distribution (Map)",
    "Popularity Levels by Rating"
]
choice = st.sidebar.radio("Select Visualization", menu)

st.title("Netflix Data Analysis")
st.write("Explore Netflix dataset with visualizations.")

# -------------------------------
# Visualizations
# -------------------------------
if choice == "Dataset Preview":
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

elif choice == "Missing Data Heatmap":
    st.subheader("Missing Data Heatmap")
    plt.figure(figsize=(10,6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    st.pyplot(plt)

elif choice == "Movies vs TV Shows":
    st.subheader("Movies vs TV Shows")
    count = df['type'].value_counts()

    plt.figure(figsize=(6,4))
    sns.barplot(
        x=count.index,
        y=count.values,
        palette=["#FF6F61", "#6BAED6"]  # two distinct colors (red & blue)
    )
    plt.title("Movies vs TV Shows", fontsize=14)
    plt.xlabel("Type")
    plt.ylabel("Count")
    st.pyplot(plt)



elif choice == "Titles Added Per Year":
    st.subheader(" Titles Added Per Year")

    # Convert 'date_added' to datetime, and extract the year
    df['year_added'] = pd.to_datetime(df['date_added'], errors='coerce').dt.year

    # Count titles added per year and sort
    year_count = df['year_added'].value_counts().sort_index()

    # Drop NaN years if any
    year_count = year_count.dropna()

    # Define colormap and normalization for color scale
    cmap = plt.cm.coolwarm  # You can try 'viridis', 'plasma', etc.
    norm = mpl.colors.Normalize(vmin=min(year_count.values), vmax=max(year_count.values))
    colors = [cmap(norm(value)) for value in year_count.values]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10,5))
    sns.barplot(x=year_count.index.astype(int), y=year_count.values, palette=colors, ax=ax)

    # Add colorbar (palette) on right-hand side
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Number of Titles', rotation=270, labelpad=15)

    # Customize appearance
    ax.set_title("Titles Added to Netflix Each Year", fontsize=14)
    ax.set_xlabel("Year Added")
    ax.set_ylabel("Number of Titles")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)

    st.pyplot(fig)
    
    



elif choice == "Titles Added Per Year":
    st.subheader("Titles Added Per Year")
    df['year_added'] = pd.to_datetime(df['date_added']).dt.year
    year_count = df['year_added'].value_counts().sort_index()
    plt.figure(figsize=(10,5))
    plt.plot(year_count.index, year_count.values, marker='o')
    plt.title("Titles Added per Year")
    st.pyplot(plt)

elif choice == "Ratings Distribution":
    st.subheader(" Rating Distribution on Netflix")

    # Clean data
    df['rating'] = df['rating'].fillna('Unknown')

    # Count ratings
    rating_counts = df['rating'].value_counts().sort_values(ascending=False)

    # Plot
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("coolwarm", len(rating_counts))

    ax = sns.barplot(x=rating_counts.index, y=rating_counts.values, palette=colors)

    # Add labels on bars
    for i, v in enumerate(rating_counts.values):
        ax.text(i, v + 10, str(v), ha='center', fontweight='bold')

    # Titles and styling
    ax.set_title("Distribution of Content Ratings on Netflix", fontsize=16, weight='bold', pad=15)
    ax.set_xlabel("Rating Category", fontsize=12)
    ax.set_ylabel("Number of Titles", fontsize=12)
    plt.xticks(rotation=45)
    ax.set_facecolor("#f9f9f9")

    # Light grid and style
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Caption
    st.markdown(" **Insight:** Ratings like *TV-MA* and *TV-14* dominate Netflix’s library, showing its focus on mature audiences.")

    # Show plot
    st.pyplot(plt)


elif choice == "Top 10 Genres":
    st.subheader(" Top 10 Genres on Netflix")

    # Prepare data
    all_genres = ', '.join(df['listed_in'].dropna()).split(', ')
    genre_count = pd.Series(all_genres).value_counts().head(10)

    # Define a color map for the bars (different color for each)
    cmap = plt.cm.viridis  # you can try 'plasma', 'coolwarm', etc.
    norm = plt.Normalize(vmin=min(genre_count.values), vmax=max(genre_count.values))
    colors = [cmap(norm(value)) for value in genre_count.values]

    # Create figure
    plt.figure(figsize=(10,5))
    sns.barplot(
        x=genre_count.values,
        y=genre_count.index,
        palette=colors
    )

    # Customize chart
    plt.title("Top 10 Netflix Genres", fontsize=14, weight='bold')
    plt.xlabel("Number of Titles")
    plt.ylabel("Genre")
    plt.grid(axis='x', linestyle='--', alpha=0.6)

    # Show in Streamlit
    st.pyplot(plt)




elif choice == "Movie Duration Distribution":
    st.subheader(" Movie Duration Distribution")

    # --- Filter only movies ---
    movies_df = df[df['type'] == 'Movie'].copy()

    # --- Clean and convert duration ---
    movies_df['duration'] = movies_df['duration'].str.replace(' min', '', regex=False)
    movies_df['duration'] = pd.to_numeric(movies_df['duration'], errors='coerce')

    # Drop missing durations
    movies_df = movies_df.dropna(subset=['duration'])

    # --- Add year range filter ---
    min_year = int(movies_df['release_year'].min())
    max_year = int(movies_df['release_year'].max())
    year_range = st.slider(
        "Select Release Year Range",
        min_year, max_year, (2010, 2021)
    )

    filtered_df = movies_df[
        (movies_df['release_year'] >= year_range[0]) & (movies_df['release_year'] <= year_range[1])
    ]

    st.write(f"🎞 Showing movie durations for **{year_range[0]} – {year_range[1]}**")

    # --- Create distribution plot (Plotly) ---
    fig = px.histogram(
        filtered_df,
        x='duration',
        nbins=30,
        color_discrete_sequence=['#E74C3C'],
        title="Distribution of Movie Durations (in minutes)",
        labels={'duration': 'Movie Duration (minutes)'},
        opacity=0.8
    )

    # Customize design
    fig.update_layout(
        bargap=0.05,
        xaxis=dict(title="Duration (minutes)", gridcolor='rgba(200,200,200,0.3)'),
        yaxis=dict(title="Number of Movies", gridcolor='rgba(200,200,200,0.3)'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title_font=dict(size=18, family='Arial Black'),
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- Summary Statistics ---
    avg_duration = int(filtered_df['duration'].mean())
    shortest = int(filtered_df['duration'].min())
    longest = int(filtered_df['duration'].max())

    st.markdown("###  Summary Insights")
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Duration", f"{avg_duration} min")
    col2.metric("Shortest Movie", f"{shortest} min")
    col3.metric("Longest Movie", f"{longest} min")

    st.markdown("""
    **Insight:**  
    Netflix's catalog typically centers around movies lasting between 90–120 minutes,  
    though a few shorter and longer outliers exist.
    """)




elif choice == "Top 10 Countries":
    st.subheader(" Top 10 Countries Producing Netflix Titles")

    # Clean and prepare country data
    df['country'] = df['country'].fillna('Unknown')
    df_countries = df['country'].dropna().str.split(',', expand=True).stack().str.strip()
    country_count = df_countries.value_counts().head(10)

    # Aesthetic setup
    plt.figure(figsize=(10, 6))
    cmap = plt.cm.plasma  # you can try 'viridis', 'coolwarm', 'magma'
    norm = plt.Normalize(vmin=min(country_count.values), vmax=max(country_count.values))
    colors = [cmap(norm(value)) for value in country_count.values]

    # Create barplot
    ax = sns.barplot(
        x=country_count.values,
        y=country_count.index,
        palette=colors
    )

    # Add count labels on bars
    for i, v in enumerate(country_count.values):
        ax.text(v + 5, i, str(v), color='black', va='center', fontweight='bold')

    # Styling and titles
    ax.set_title("Top 10 Countries with Most Netflix Titles", fontsize=16, weight='bold', pad=15)
    ax.set_xlabel("Number of Titles", fontsize=12)
    ax.set_ylabel("Country", fontsize=12)
    ax.grid(axis='x', linestyle='--', alpha=0.3)

    # Add subtle background color and remove borders
    ax.set_facecolor("#f9f9f9")
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add small color description
    st.markdown("**Color intensity** represents how prolific a country is on Netflix.")

    # Show in Streamlit
    st.pyplot(plt)


elif choice == "Correlation Heatmap":
    st.subheader("Correlation Heatmap")
    df['year_added'] = pd.to_datetime(df['date_added'], errors='coerce').dt.year
    df_numeric = df[['release_year', 'year_added']].dropna()
    corr = df_numeric.corr()
    plt.figure(figsize=(5,4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    st.pyplot(plt)


    
elif choice == "Popularity Levels by Rating":
    st.subheader("⭐ Popularity Levels by Rating")

    # Define custom mapping
    popularity_map = {
        'TV-14': 'Popular',
        'PG-13': 'Popular',
        'TV-PG': 'Popular',
        'R': 'Average',
        'TV-MA': 'Average',
        'PG': 'Popular',
        'TV-Y7': 'Not Popular',
        'TV-Y7-FV': 'Not Popular',
        'G': 'Not Popular',
        'TV-G': 'Not Popular',
        'NC-17': 'Average',
        'NR': 'Not Popular',
        'Unknown': 'Not Popular'
    }

    # Apply mapping
    df['popularity'] = df['rating'].map(popularity_map).fillna('Not Popular')

    # Count how many shows fall into each popularity level
    pop_count = df['popularity'].value_counts()

    # Visualization
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(
        x=pop_count.index,
        y=pop_count.values,
        palette=['#2ecc71', '#f39c12', '#e74c3c'],  # green, orange, red
        ax=ax
    )

    ax.set_title("Distribution of Netflix Titles by Popularity Level", fontsize=14)
    ax.set_xlabel("Popularity Category")
    ax.set_ylabel("Number of Titles")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Display chart
    st.pyplot(fig)

    # Show table preview
    with st.expander("📋 View Sample Data"):
        st.dataframe(df[['title', 'rating', 'popularity']].head(10))

    # Show count summary
    st.write("### 📈 Count by Popularity Level")
    st.write(pop_count)





elif choice == "Global Distribution (Map)":
    st.subheader("🌎 Global Distribution of Netflix Titles")

    # Clean and prepare data
    df['country'] = df['country'].fillna('Unknown')

    # Split multiple countries (e.g., "United States, Canada")
    df_countries = df['country'].dropna().str.split(',', expand=True).stack().str.strip()
    country_count = df_countries.value_counts().reset_index()
    country_count.columns = ['Country', 'Count']

    # Build interactive world map
    fig = go.Figure(
        data=go.Choropleth(
            locations=country_count['Country'],
            locationmode='country names',
            z=country_count['Count'],
            colorscale='YlOrRd',
            colorbar_title='Number of Titles',
            marker_line_color='white',
            hovertemplate='<b>%{location}</b><br>Titles: %{z}<extra></extra>'  # nice hover info
        )
    )

    fig.update_layout(
        title_text="🌍 Netflix Titles by Country",
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="black",
            showland=True,
            landcolor="coral",
            showocean=True,
            oceancolor="aqua",
            projection_type="equirectangular"
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )

    # Show map in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Optional summary bar chart
    st.subheader("Top 10 Countries by Number of Titles")
    st.bar_chart(country_count.head(10).set_index('Country'))


elif choice == "TV Show Season Counts":
    st.subheader(" TV Show Season Counts")

    # --- Filter TV Shows only ---
    tv_df = df[df['type'] == 'TV Show'].copy()

    # --- Clean and convert 'duration' column to numeric (number of seasons) ---
    tv_df['seasons'] = tv_df['duration'].str.replace(' Season', '', regex=False)
    tv_df['seasons'] = tv_df['seasons'].str.replace('s', '', regex=False)
    tv_df['seasons'] = pd.to_numeric(tv_df['seasons'], errors='coerce')

    # Drop invalid values
    tv_df = tv_df.dropna(subset=['seasons'])

    # --- Range slider to filter by season count ---
    min_seasons = int(tv_df['seasons'].min())
    max_seasons = int(tv_df['seasons'].max())

    season_range = st.slider(
        "Select Number of Seasons Range",
        min_seasons, max_seasons, (1, 5)
    )

    filtered_df = tv_df[
        (tv_df['seasons'] >= season_range[0]) & (tv_df['seasons'] <= season_range[1])
    ]

    st.write(f"Showing TV Shows with **{season_range[0]} to {season_range[1]}** seasons")

    # --- Distribution Bar Chart ---
    fig_bar = px.histogram(
        filtered_df,
        x='seasons',
        nbins=max_seasons,
        color_discrete_sequence=['#3498DB'],
        title="Distribution of TV Show Season Counts",
        labels={'seasons': 'Number of Seasons'},
        opacity=0.8
    )

    fig_bar.update_layout(
        bargap=0.05,
        xaxis=dict(title="Number of Seasons", gridcolor='rgba(200,200,200,0.3)'),
        yaxis=dict(title="Number of TV Shows", gridcolor='rgba(200,200,200,0.3)'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title_font=dict(size=18, family='Arial Black'),
    )

    st.plotly_chart(fig_bar, use_container_width=True)

    # --- Pie Chart for Proportion ---
    st.markdown("### Proportion of TV Shows by Season Count")
    season_count = filtered_df['seasons'].value_counts().sort_index()
    fig_pie = px.pie(
        names=season_count.index.astype(str),
        values=season_count.values,
        color_discrete_sequence=px.colors.sequential.Tealgrn,
        title="TV Show Season Distribution (Filtered Range)"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # --- Summary Metrics ---
    avg_seasons = round(filtered_df['seasons'].mean(), 1)
    most_common = int(filtered_df['seasons'].mode()[0])
    max_show = int(filtered_df['seasons'].max())

    st.markdown("### Quick Insights")
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Seasons", f"{avg_seasons}")
    col2.metric("Most Common", f"{most_common} seasons")
    col3.metric("Longest Series", f"{max_show} seasons")

    st.markdown("""
    💡 **Insight:**  
    Most Netflix TV shows have **1–3 seasons**, suggesting short-format series dominate the platform.
    """)

elif choice == "Data Cleaning":
    st.subheader(" Data Cleaning Process")

    # 1️⃣ Show missing values count
    st.write("###  Missing Values Before Cleaning")
    missing_before = df.isnull().sum()
    st.dataframe(missing_before[missing_before > 0])

    # 2️⃣ Fill or drop missing data intelligently
    st.write("### 🛠 Cleaning Steps")
    st.markdown("""
    - Filled missing **country** values with 'Unknown'  
    - Filled missing **rating** values with 'Unknown'  
    - Dropped rows where **date_added** is missing (since it's essential for time analysis)  
    - Trimmed whitespace in string columns  
    """)

    # Make a copy to clean
    df_cleaned = df.copy()

    # Fill missing values
    df_cleaned['country'] = df_cleaned['country'].fillna('Unknown')
    df_cleaned['rating'] = df_cleaned['rating'].fillna('Unknown')

    # Drop rows missing critical date
    df_cleaned = df_cleaned.dropna(subset=['date_added'])

    # Clean whitespace in all string columns
    str_cols = df_cleaned.select_dtypes(include='object').columns
    df_cleaned[str_cols] = df_cleaned[str_cols].apply(lambda x: x.str.strip())

    # 3️⃣ Show missing values after cleaning
    st.write("### Missing Values After Cleaning")
    missing_after = df_cleaned.isnull().sum()
    st.dataframe(missing_after[missing_after > 0])

    # 4️⃣ Summary of cleaning
    st.success(f"Data cleaned successfully! Rows before: {len(df)}, after cleaning: {len(df_cleaned)}")

    # 5️⃣ Show preview of cleaned data
    st.write("### Cleaned Data Preview")
    st.dataframe(df_cleaned.head(10))

    # (Optional) allow download of cleaned data
    csv = df_cleaned.to_csv(index=False).encode('utf-8')
    st.download_button(" Download Cleaned Dataset (CSV)", csv, "netflix_cleaned.csv", "text/csv")



elif choice == "Top 10 Actors":
    st.subheader(" Top 10 Most Featured Actors on Netflix")

    # --- Data Preparation ---
    actor_series = df['cast'].dropna().str.split(',').explode().str.strip()
    actor_count = actor_series.value_counts().head(10)

    # --- Optional Search ---
    st.markdown("### Search for an Actor")
    actor_search = st.text_input("Enter an actor's name (case-insensitive):")

    if actor_search:
        matched = actor_series[actor_series.str.contains(actor_search, case=False)]
        if len(matched) > 0:
            st.success(f"Found {len(matched)} appearance(s) for '{actor_search}'.")
        else:
            st.warning(f" No results found for '{actor_search}'.")

    # --- Bar Chart (Top 10 Actors) ---
    fig_bar = px.bar(
        x=actor_count.values,
        y=actor_count.index,
        orientation='h',
        color=actor_count.values,
        color_continuous_scale='tealgrn',
        labels={'x': 'Number of Titles', 'y': 'Actor'},
        title='Top 10 Actors with the Most Netflix Appearances'
    )

    fig_bar.update_layout(
        xaxis_title="Number of Titles",
        yaxis_title="Actor",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title_font=dict(size=18, family='Arial Black'),
        height=500
    )

    st.plotly_chart(fig_bar, use_container_width=True)

    # --- Bubble Chart (Actor Popularity Visualization) ---
    st.markdown("### Actor Popularity Bubble Chart")

    actor_df = pd.DataFrame({
        'Actor': actor_count.index,
        'Appearances': actor_count.values
    })

    fig_bubble = px.scatter(
        actor_df,
        x='Actor',
        y='Appearances',
        size='Appearances',
        color='Appearances',
        color_continuous_scale='Viridis',
        hover_name='Actor',
        size_max=60,
        title="Actor Popularity Visualization (Bubble Size = No. of Titles)"
    )

    fig_bubble.update_layout(
        xaxis_title="Actor",
        yaxis_title="Number of Titles",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        title_font=dict(size=16, family='Arial Black'),
    )

    st.plotly_chart(fig_bubble, use_container_width=True)

    # --- Quick Insights ---
    st.markdown("###  Quick Insights")
    col1, col2, col3 = st.columns(3)
    col1.metric("Most Featured Actor", actor_count.index[0])
    col2.metric("Titles Appeared In", actor_count.iloc[0])
    col3.metric("Unique Actors", actor_series.nunique())

    st.markdown("""
     **Insight:**  
    Most frequent Netflix actors often appear across multiple genres — 
    showing the platform's trend of casting familiar global faces in popular titles.
    """)


elif choice == "Top 10 Directors":
    st.subheader(" Top 10 Most Active Directors on Netflix")

    # --- Data Preparation ---
    director_series = df['director'].dropna().str.split(',').explode().str.strip()
    director_count = director_series.value_counts().head(10)

    # --- Optional Search ---
    st.markdown("### Search for a Director")
    director_search = st.text_input("Enter a director's name (case-insensitive):")

    if director_search:
        matched = director_series[director_series.str.contains(director_search, case=False)]
        if len(matched) > 0:
            st.success(f" Found {len(matched)} title(s) directed by '{director_search}'.")
        else:
            st.warning(f"No results found for '{director_search}'.")

    # --- Bar Chart: Top 10 Directors ---
    fig_bar = px.bar(
        x=director_count.values,
        y=director_count.index,
        orientation='h',
        color=director_count.values,
        color_continuous_scale='magma',
        labels={'x': 'Number of Titles', 'y': 'Director'},
        title="Top 10 Directors with the Most Netflix Titles"
    )

    fig_bar.update_layout(
        xaxis_title="Number of Titles",
        yaxis_title="Director",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title_font=dict(size=18, family='Arial Black'),
        height=500
    )

    st.plotly_chart(fig_bar, use_container_width=True)

    # --- Pie Chart: Country Distribution for Top Directors ---
    st.markdown("###  Country Distribution of Titles Directed by Top 10 Directors")

    # Map directors to their respective countries (based on df)
    director_country_df = df[['director', 'country']].dropna()
    director_country_df = director_country_df[
        director_country_df['director'].isin(director_count.index)
    ]

    # Explode directors with multiple countries
    director_country_df['country'] = director_country_df['country'].str.split(', ')
    director_country_df = director_country_df.explode('country')

    # Count countries
    country_count = director_country_df['country'].value_counts().head(10)

    fig_pie = px.pie(
        names=country_count.index,
        values=country_count.values,
        color_discrete_sequence=px.colors.sequential.RdBu,
        title="Top Countries Represented by These Directors"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # --- Quick Insights ---
    st.markdown("### Quick Insights")
    col1, col2, col3 = st.columns(3)
    col1.metric("Most Active Director", director_count.index[0])
    col2.metric("Titles Directed", director_count.iloc[0])
    col3.metric("Unique Directors", director_series.nunique())

    st.markdown("""
     **Insight:**  
    Netflix collaborates with directors from **diverse global regions**, 
    but the majority of frequent contributors come from the **US, India, and UK**.  
    This reflects Netflix's strategy of building regionally strong yet globally appealing catalogs.
    """)

elif choice == "User Choice Analysis":
    st.subheader(" Explore Any Netflix Data Column")

    # --- Step 1: Let user pick a column ---
    columns_to_explore = [
        'type', 'country', 'rating', 'listed_in', 'director', 'cast', 'release_year'
    ]
    selected_col = st.selectbox("Select a column to analyze:", columns_to_explore)

    # --- Step 2: Data cleaning based on column type ---
    df_selected = df[selected_col].dropna()

    # Handle multi-value fields like cast, director, genres
    if selected_col in ['listed_in', 'cast', 'director']:
        df_selected = df_selected.str.split(',').explode().str.strip()

    # --- Step 3: Calculate value counts ---
    value_counts = df_selected.value_counts().head(15)
    total_unique = df_selected.nunique()

    # --- Step 4: Visualization type selection ---
    viz_type = st.radio(
        " Choose a visualization type:",
        ["Bar Chart", "Pie Chart", "Treemap"]
    )

    # --- Step 5: Generate visualization ---
    st.markdown(f"### 🔍 Top 15 Values for **{selected_col.capitalize()}**")

    if viz_type == "Bar Chart":
        fig = px.bar(
            x=value_counts.values,
            y=value_counts.index,
            orientation='h',
            color=value_counts.values,
            color_continuous_scale='Viridis',
            labels={'x': 'Count', 'y': selected_col.capitalize()},
            title=f"Top {len(value_counts)} {selected_col.capitalize()}s"
        )
        fig.update_layout(
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font=dict(size=18, family='Arial Black')
        )
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Pie Chart":
        fig = px.pie(
            names=value_counts.index,
            values=value_counts.values,
            title=f"Distribution of {selected_col.capitalize()}",
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        fig.update_traces(textinfo='percent+label')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Treemap":
        fig = px.treemap(
            names=value_counts.index,
            parents=[""] * len(value_counts),
            values=value_counts.values,
            title=f"Treemap of {selected_col.capitalize()}"
        )
        fig.update_layout(
            height=500,
            margin=dict(t=50, l=0, r=0, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Step 6: Quick Insights ---
    st.markdown("### Quick Insights")
    col1, col2, col3 = st.columns(3)
    col1.metric("Unique Values", total_unique)
    col2.metric("Top Category", value_counts.index[0])
    col3.metric("Top Count", int(value_counts.iloc[0]))

    # --- Step 7: Search Feature ---
    st.markdown("###  Search Within This Column")
    query = st.text_input("Enter keyword to search:")

    if query:
        matched = df[df[selected_col].astype(str).str.contains(query, case=False, na=False)]
        count_matches = len(matched)

        if count_matches > 0:
            st.success(f" Found {count_matches} entries containing '{query}' in column **{selected_col}**.")
            st.dataframe(matched[[selected_col, 'title']].head(10))  # show top 10 results
            if count_matches > 10:
                st.info(f"Showing first 10 of {count_matches} matches.")
        else:
            st.warning(f" No results found for '{query}'.")


else:
    st.info("Visualization under development. Please check back later!")




