# Game Rating & Critic Analysis Dashboard

This project is a sophisticated web application built with Streamlit for a group of critics to rate and analyze video games. It moves beyond simple averages by implementing custom statistical models to provide more robust game rankings and to quantify how "controversial" a critic's opinions are compared to the group consensus.

The application is designed to address common problems in rating systems with sparse data, such as when not every critic rates every single game.

## ✨ Core Features

* **Secure Login:** The dashboard is private and protected by a password.
* **Dashboard View:** Features Key Performance Indicators (KPIs) like total ratings, overall average score, and group participation. It also includes a main game leaderboard and visual breakdowns of critic nominations and score distributions.
* **Detailed Game Analysis:** Each game has a dedicated page showing:
    * Both its **raw average score** and a statistically **adjusted final score**.
    * Metrics like "Play Rate" (what percentage of critics rated it) and "Controversy" (the standard deviation of its scores).
    * A transparent, step-by-step breakdown of how the adjusted score is calculated using a custom Bayesian average.
* **In-Depth Critic Profiles:** Each critic has a page that analyzes their rating habits, including:
    * A scorecard with their personal participation rate and average score.
    * A unique **"Controversy Score"** that measures how much their ratings and participation habits deviate from the group consensus.
    * Detailed tables showing their most and least "contrarian" ratings.

---

## 🧐 Key Concepts Explained

The core of this application lies in two custom statistical models designed to create a fairer and more insightful analysis.

### 1. Adjusted Game Ranking (Bayesian Average)

A common problem in ranking systems is how to compare an item with few ratings to one with many. This app solves the problem by using a form of **Bayesian averaging** that "shrinks" a game's score toward a prior belief.

* **The Model:** The final adjusted score is a weighted average of the game's raw average score and a "pessimistic prior."
* **Pessimistic Prior:** For any given game, a "prior" score is calculated based on the personal statistics of every critic who *chose not to rate it*. This prior is calculated as `Critic's Average Score - Critic's Standard Deviation`.
* **The Effect:** A game with very few ratings (`n`) will have its final score pulled heavily towards this pessimistic prior. A game with many ratings will have its score determined almost entirely by its own raw average. This provides a more stable and reliable ranking.

The formula used is displayed on the Game Details page:
$$\text{Final Score} = \frac{(n \times \text{Raw Avg}) + (C \times \text{Pessimistic Prior})}{(n + C)}$$
Where:
* $n$ = Number of critics who rated the game.
* $C$ = Number of critics who skipped the game.

### 2. Critic Controversy Score

This score is a custom metric designed to identify which critics have the most unique or contrarian tastes compared to the group. It is also adjusted using Bayesian shrinkage to ensure critics with more ratings have more credible scores.

The score is composed of two main factors:

1.  **Score Deviation:** Measures how far, on average, a critic's score for a game deviates from that game's group average.
2.  **Play Deviation:** A unique metric that measures how often a critic's *decision to rate or skip a game* goes against the grain.
    * A critic is rewarded for rating an unpopular (low participation) game.
    * A critic is also flagged for *not* rating a very popular (high participation) game.

These two factors are combined into an "Observed Score," which is then adjusted using Bayesian shrinkage toward the group's average controversy score. A critic's final score is given more "credibility" based on the number of games they have rated.

---

## 🛠️ Technology Stack

* **Backend:** Python
* **Web Framework:** Streamlit
* **Data Manipulation:** Pandas
* **Database ORM:** SQLAlchemy
* **Data Visualization:** Plotly
* **Database:** (User's choice, e.g., PostgreSQL, SQLite)

---

## 🚀 Setup and Installation

Follow these steps to get the application running locally.

### 1. Prerequisites

* Python 3.8+
* An active database instance (e.g., PostgreSQL).

### 2. Clone the Repository

```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
````

### 3\. Set Up a Virtual Environment

It is highly recommended to use a virtual environment.

```bash
# For Mac/Linux
python3 -m venv .venv
source .venv/bin/activate

# For Windows
python -m venv .venv
.venv\Scripts\activate
```

### 4\. Install Dependencies

Create a `requirements.txt` file with the necessary libraries and install them.

**`requirements.txt`:**

```
streamlit
pandas
sqlalchemy
plotly
psycopg2-binary # Or the appropriate driver for your database
```

**Installation Command:**

```bash
pip install -r requirements.txt
```

### 5\. Configure Secrets

The application uses Streamlit's secrets management. Create a file named `.streamlit/secrets.toml` in the project's root directory.

**`.streamlit/secrets.toml`:**

```toml
# The password to access the dashboard
password = "your_secret_password"

# Database connection details
[connections.mydb]
dialect = "postgresql"
host = "your_db_host"
port = 5432
database = "your_db_name"
username = "your_db_username"
password = "your_db_password"
```

### 6\. Initialize the Database

Run a script or manually use an SQL client to create the database tables based on the schema defined in `database_models.py`.

### 7\. Run the Application

```bash
streamlit run streamlit_app.py
```

The application should now be running and accessible in your web browser\!

-----

## 📂 File Structure

A brief overview of the key files in this project:

```
.
├── .streamlit/
│   └── secrets.toml        # Holds passwords and connection strings
├── pages/
│   ├── dashboard.py        # UI and logic for the main dashboard page
│   ├── critic_details.py   # UI and logic for the critic analysis page
│   ├── game_details.py     # UI and logic for the game details page
│   └── predictions.py      # Placeholder for future predictive models
├── streamlit_app.py        # Main entry point, handles login and navigation
├── utils.py                # Core calculation logic for rankings and controversy
├── database_models.py      # SQLAlchemy ORM models for the database schema
├── queries.json            # (Legacy) Raw SQL queries
└── requirements.txt        # Python package dependencies
```

## 🔮 Future Work

As indicated in `pages/predictions.py`, future development could include:

  * Predicting a score for any critic on any game (including upcoming ones).
  * Ranking the features that contribute to each prediction.
  * Forecasting the likelihood a critic will skip a particular game.

-----

## 📜 License

This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.
