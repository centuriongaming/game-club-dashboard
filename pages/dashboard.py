# pages/dashboard.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sqlalchemy as sa
import plotly.figure_factory as ff
from utils import check_auth, get_sqla_session, calculate_controversy_scores, calculate_custom_game_rankings
from database_models import Critic, Game, Rating

# --- Page & Data Configuration ---
st.set_page_config(page_title="Dashboard", layout="wide")

@st.cache_data
def load_dashboard_data(_session):
    """Runs all expensive queries and calculations for the dashboard at once."""
    critics_df = pd.read_sql(sa.select(Critic.id, Critic.critic_name).order_by(Critic.critic_name), _session.bind)
    games_df = pd.read_sql(sa.select(Game.id, Game.game_name).where(Game.upcoming == False), _session.bind)
    full_ratings_scaffold_df = pd.read_sql(sa.select(Rating.critic_id, Rating.game_id, Rating.score), _session.bind)
    ratings_df = full_ratings_scaffold_df.dropna(subset=['score']).copy()

    rankings_df, _ = calculate_custom_game_rankings(games_df, critics_df, ratings_df)
    
    game_controversy = ratings_df.groupby('game_id')['score'].std().reset_index()
    game_controversy = game_controversy.rename(columns={'score': 'controversy'})
    game_controversy['controversy'] = game_controversy['controversy'].fillna(0)

    rankings_df = pd.merge(rankings_df, game_controversy, left_on='id', right_on='game_id', how='left')
    rankings_df['controversy'] = rankings_df['controversy'].fillna(0)
    
    critic_names = critics_df['critic_name'].tolist()

    total_ratings = len(ratings_df)
    group_participation = (total_ratings / len(full_ratings_scaffold_df)) * 100 if not full_ratings_scaffold_df.empty else 0
    kpis = {
        "total_ratings": total_ratings,
        "avg_score": ratings_df['score'].mean() if not ratings_df.empty else 0,
        "group_participation": group_participation,
        "ratings_df": ratings_df
    }

    nomination_stmt = sa.select(Critic.critic_name, sa.func.count(Game.id).label("nomination_count")).join(Game, Critic.id == Game.nominated_by, isouter=True).group_by(Critic.critic_name)
    nomination_df = pd.read_sql(nomination_stmt, _session.bind).set_index('critic_name').reindex(critic_names).fillna(0).reset_index()

    binned_stmt = sa.select(Critic.critic_name, sa.func.floor(Rating.score).label("score_bin"), sa.func.count(Rating.id).label("rating_count")).join(Rating, Critic.id == Rating.critic_id).where(Rating.score.is_not(None)).group_by(Critic.critic_name, "score_bin")
    binned_df = pd.read_sql(binned_stmt, _session.bind)
    
    participation_stmt = sa.select(Critic.critic_name, sa.func.count(Rating.score).label("ratings_given"), sa.func.avg(Rating.score).label("average_score")).join(Rating, Critic.id == Rating.critic_id, isouter=True).group_by(Critic.critic_name).order_by(sa.desc("ratings_given"))
    critic_participation_df = pd.read_sql(participation_stmt, _session.bind)
    if not games_df.empty:
        critic_participation_df['participation_rate'] = (critic_participation_df['ratings_given'] / len(games_df)) * 100

    upcoming_stmt = sa.select(Game.game_name, Critic.critic_name.label("nominated_by")).join(Critic, Game.nominated_by == Critic.id, isouter=True).where(Game.upcoming == True).order_by(Game.game_name)
    upcoming_games_df = pd.read_sql(upcoming_stmt, _session.bind)

    return kpis, rankings_df, nomination_df, binned_df, critic_participation_df, upcoming_games_df, critic_names

# --- UI Component Functions ---
def display_kpis(kpis):
    """Display the key performance indicators."""
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Ratings Given", f"{kpis['total_ratings']}")
        col2.metric("Overall Average Score", f"{kpis['avg_score']:.2f}" if kpis['avg_score'] else "N/A")
        col3.metric("Group Participation", f"{kpis['group_participation']:.1f}%")

def display_game_showcase(rankings_df, kpis):
    """Display the top/bottom games and the new plots."""
    with st.container(border=True):
        st.subheader("Game Leaderboard & Analysis")
        if rankings_df.empty:
            st.info("No game rankings to display yet.")
            return

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**First**: {rankings_df.iloc[0]['game_name']}")
        with col2:
            st.markdown(f"**Last**: {rankings_df.iloc[-1]['game_name']} (#{rankings_df.iloc[-1]['Rank']})")
        
        st.markdown("---")

        plot_tab1, plot_tab2 = st.tabs(["Score vs. Controversy Plot", "Overall Score Distribution"])

        with plot_tab1:
            display_score_vs_controversy_plot(rankings_df)

        with plot_tab2:
            display_overall_score_distribution(kpis['ratings_df'], kpis['avg_score'])

def display_score_vs_controversy_plot(rankings_df):
    """Displays an interactive scatter plot of game scores vs. controversy."""
    if rankings_df.empty or 'controversy' not in rankings_df:
        st.info("Not enough data to generate plot.")
        return

    fig = px.scatter(
        rankings_df,
        x="final_adjusted_score",
        y="controversy",
        size="number_of_ratings",
        color="average_score",
        color_continuous_scale=px.colors.sequential.Viridis,
        hover_name="game_name",
        hover_data=['Rank', 'number_of_ratings', 'average_score'],
        labels={
            "final_adjusted_score": "Final Adjusted Score (Higher is Better)",
            "controversy": "Controversy (Score Standard Deviation)",
            "number_of_ratings": "# of Ratings",
            "average_score": "Raw Avg Score"
        },
        title="Game Score vs. Controversy"
    )
    fig.update_layout(xaxis_title="Final Adjusted Score",
                      yaxis_title="Controversy (Std. Dev.)",
                      yaxis_range=[0, 10] # <-- ADD THIS LINE
                      )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Hover over a point to see game details. Bubble size represents the number of ratings.")

def display_overall_score_distribution(ratings_df, avg_score):
    """Displays a KDE plot of all scores given by the group."""
    if ratings_df.empty or ratings_df['score'].nunique() < 2:
        st.info("At least two unique scores are needed to generate a distribution plot.")
        return

    fig = ff.create_distplot(
        [ratings_df['score']],
        ['Overall Scores'],
        bin_size=.25,
        show_hist=False,
        show_rug=False
    )
    fig.add_vline(x=avg_score, line_width=2, line_dash="dash", line_color="red", annotation_text=f"Mean: {avg_score:.2f}")
    fig.update_layout(
        title_text='Group Rating Density (KDE)',
        xaxis_title="Score",
        yaxis_title="Density",
        showlegend=False,
        xaxis_range=[0, 10] # <-- ADD THIS LINE TO SET X-AXIS
    )
    st.plotly_chart(fig, use_container_width=True)

def display_critic_visuals(nomination_df, binned_df, critic_names, color_map):
    """Display the pie and bar charts for critic analysis."""
    st.subheader("Visual Breakdowns")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Nominations by Critic")
        pie_colors = [color_map.get(name) for name in nomination_df['critic_name']]
        fig = go.Figure(data=[go.Pie(labels=nomination_df['critic_name'], values=nomination_df['nomination_count'], hole=.3, textinfo='label+percent', marker=dict(colors=pie_colors))])
        fig.update_layout(showlegend=False, height=350, margin=dict(l=1, r=1, t=1, b=1))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("##### Score Distribution by Critic")
        if not binned_df.empty:
            binned_pivot = binned_df.pivot(index='score_bin', columns='critic_name', values='rating_count').reindex(columns=critic_names, fill_value=0)
            bar_chart_colors = [color_map.get(name) for name in binned_pivot.columns]
            st.bar_chart(binned_pivot, height=310, color=bar_chart_colors)

def display_controversy_table(session):
    """Display the ranked table of critic controversy scores."""
    st.subheader("Critic Controversy Ranking")
    st.caption("A statistically-adjusted score representing how a critic's rating and participation deviates from the group consensus.")
    
    critic_controversy_df, _ = calculate_controversy_scores(session)
    
    if critic_controversy_df.empty:
        st.info("Not enough data to calculate controversy scores.")
        return
        
    sorted_df = critic_controversy_df.sort_values('final_controversy_score', ascending=False)
    sorted_df = sorted_df.reset_index(drop=True)
    sorted_df['Rank'] = sorted_df.index + 1
    
    st.dataframe(
        sorted_df[['Rank', 'critic_name', 'final_controversy_score']],
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", format="%d"), 
            "critic_name": "Critic",
            "final_controversy_score": st.column_config.NumberColumn("Final Controversy Score", format="%.3f")
        },
        hide_index=True, use_container_width=True
    )

# --- Main Page ---
def main():
    """Renders the main dashboard page."""
    check_auth()
    session = get_sqla_session()
    
    st.title("Dashboard & Leaderboards")

    data = load_dashboard_data(session)
    kpis, rankings_df, nomination_df, binned_df, critic_participation_df, upcoming_games_df, critic_names = data

    colors = px.colors.qualitative.Plotly
    color_map = {name: colors[i % len(colors)] for i, name in enumerate(critic_names)}

    display_kpis(kpis)
    display_game_showcase(rankings_df, kpis)

    tab1, tab2, tab3 = st.tabs(["Game Rankings", "Critic Analysis", "Upcoming Games"])

    with tab1:
        st.subheader("Full Game Rankings")
        columns_to_show = ['Rank', 'game_name', 'final_adjusted_score', 'number_of_ratings']
        st.dataframe(
            rankings_df[columns_to_show],
            column_config={
                "Rank": "Rank",
                "game_name": "Game",
                "final_adjusted_score": st.column_config.NumberColumn("Score", format="%.3f"),
                "number_of_ratings": "# Ratings",
            },
            hide_index=True, use_container_width=True,
        )

    with tab2:
        display_critic_visuals(nomination_df, binned_df, critic_names, color_map)
        
        st.subheader("Critics by Participation")
        st.dataframe(
            critic_participation_df,
            column_config={
                "critic_name": "Critic",
                "ratings_given": "Ratings Given",
                "average_score": st.column_config.ProgressColumn("Average Score", format="%.2f", min_value=0, max_value=10),
                "participation_rate": st.column_config.ProgressColumn("Participation", format="%.1f%%", min_value=0, max_value=100)
            },
            hide_index=True, use_container_width=True
        )
        
        display_controversy_table(session)

    with tab3:
        st.dataframe(
            upcoming_games_df.rename(columns={"game_name": "Game", "nominated_by": "Nominated By"}),
            hide_index=True, use_container_width=True
        )
    
    if st.button("Log out"):
        st.session_state["password_correct"] = False
        st.rerun()

if __name__ == "__main__":
    main()