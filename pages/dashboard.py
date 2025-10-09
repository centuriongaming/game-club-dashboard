# pages/dashboard.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils import check_auth, get_db_connection, load_queries

# --- Initial Setup ---
check_auth()
conn = get_db_connection()
queries = load_queries()

# --- Page Configuration ---
st.set_page_config(page_title="Dashboard", layout="wide")
st.title("Dashboard & Leaderboards")

# --- Establish Consistent Critic Order and Color Map ---
critics_list_df = conn.query(queries['get_critics_list'])
critic_names = critics_list_df['critic_name'].tolist()
colors = px.colors.qualitative.Plotly
color_map = {name: colors[i % len(colors)] for i, name in enumerate(critic_names)}

# --- Key Performance Indicators ---
with st.container(border=True):
    total_ratings = conn.query(queries['get_kpi_total_ratings'])['total'][0]
    avg_score = conn.query(queries['get_kpi_avg_score'])['average'][0]
    participation_rate = conn.query(queries['get_kpi_participation'])['rate'][0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Ratings Given", f"{total_ratings}")
    col2.metric("Overall Average Score", f"{avg_score:.2f}")
    col3.metric("Group Participation", f"{participation_rate:.1f}%")

# --- Main Data Query and Game Ranking Calculation ---
rankings_df = conn.query(queries['get_game_rankings'])
rankings_df = rankings_df.sort_values("final_adjusted_score", ascending=False).reset_index(drop=True)
rankings_df['Rank'] = rankings_df.index + 1
rankings_df['Unadjusted Rank'] = rankings_df['average_score'].rank(method='min', ascending=False).astype(int)

# --- Top & Bottom Ranked Games Showcase ---
with st.container(border=True):
    st.subheader("Top & Bottom Ranked Games")
    best_adjusted = rankings_df.loc[rankings_df['Rank'].idxmin()]
    worst_adjusted = rankings_df.loc[rankings_df['Rank'].idxmax()]
    best_unadjusted = rankings_df.loc[rankings_df['Unadjusted Rank'].idxmin()]
    worst_unadjusted = rankings_df.loc[rankings_df['Unadjusted Rank'].idxmax()]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Final Ranking")
        st.markdown(f"**#1**: {best_adjusted['game_name']}")
        st.markdown(f"**Last**: {worst_adjusted['game_name']} (#{worst_adjusted['Rank']})")
    with col2:
        st.markdown("##### Unadjusted Ranking")
        st.markdown(f"**#1**: {best_unadjusted['game_name']}")
        st.markdown(f"**Last**: {worst_unadjusted['game_name']} (#{worst_unadjusted['Unadjusted Rank']})")


# --- Tabs for Main Content ---
tab1, tab2, tab3 = st.tabs(["Game Rankings", "Critic Analysis", "Upcoming Games"])

# --- Game Rankings Tab ---
with tab1:
    st.dataframe(
        rankings_df[['Rank', 'game_name', 'Unadjusted Rank', 'number_of_ratings']],
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "game_name": "Game",
            "Unadjusted Rank": "Unadjusted Rank",
            "number_of_ratings": "Ratings"
        },
        hide_index=True,
        use_container_width=True
    )

# --- Critic Analysis Tab ---
with tab2:
    st.subheader("Visual Breakdowns")
    col1, col2 = st.columns(2)
    with col1:
        nomination_df = conn.query(queries['get_critic_nominations'])
        nomination_df = nomination_df.set_index('critic_name').reindex(critic_names).fillna(0).reset_index()
        pie_colors = [color_map[name] for name in nomination_df['critic_name']]
        
        fig = go.Figure(data=[go.Pie(
            labels=nomination_df['critic_name'],
            values=nomination_df['nomination_count'],
            hole=.3,
            textinfo='label+percent',
            marker=dict(colors=pie_colors)
        )])
        fig.update_layout(title_text="Nominations by Critic", showlegend=False, height=350, margin=dict(l=1, r=1, t=30, b=1))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        binned_df = conn.query(queries['get_critic_score_distribution'])
        if not binned_df.empty:
            binned_pivot = binned_df.pivot(index='score_bin', columns='critic_name', values='rating_count').fillna(0)
            binned_pivot = binned_pivot.reindex(columns=critic_names, fill_value=0)
            bar_chart_colors = [color_map[name] for name in binned_pivot.columns]
            
            st.markdown("##### Score Distribution by Critic")
            st.bar_chart(binned_pivot, height=310, color=bar_chart_colors)

    st.subheader("Critics by Participation")
    critic_participation_df = conn.query(queries['get_critic_participation'])
    critic_participation_df['participation_rate'] = (critic_participation_df['ratings_given'] / critic_participation_df['total_games']) * 100
    st.dataframe(
        critic_participation_df[['critic_name', 'ratings_given', 'participation_rate', 'average_score']],
        column_config={
            "critic_name": "Critic",
            "ratings_given": "Ratings Given",
            "average_score": st.column_config.ProgressColumn("Average Score",format="%.2f",min_value=0,max_value=10),
            "participation_rate": st.column_config.ProgressColumn("Participation",format="%.1f%%",min_value=0,max_value=100)
        },
        hide_index=True, use_container_width=True
    )
    
    st.subheader("Critic Controversy Ranking")
    critic_controversy_df = conn.query(queries['get_critic_controversy'])
    critic_controversy_df['Rank'] = critic_controversy_df['controversy_score'].rank(method='min', ascending=False).astype(int)
    
    st.dataframe(
        critic_controversy_df[['Rank', 'critic_name', 'controversy_score']],
        column_config={
            "Rank": "Rank",
            "critic_name": "Critic",
            "controversy_score": st.column_config.NumberColumn("Controversy Score", format="%.3f")
        },
        hide_index=True,
        use_container_width=True
    )

# --- Upcoming Games Tab ---
with tab3:
    upcoming_games_df = conn.query(queries['get_upcoming_games'])
    if upcoming_games_df.empty:
        st.info("There are currently no games marked as upcoming.")
    else:
        st.dataframe(
            upcoming_games_df.rename(columns={"game_name": "Game", "nominated_by": "Nominated By"}),
            hide_index=True, use_container_width=True
        )

# --- Log Out Button ---
if st.button("Log out"):
    st.session_state["password_correct"] = False
    st.rerun()