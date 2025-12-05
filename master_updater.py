import pandas as pd
import numpy as np
import requests
import json
import time
import os
import ast
import sys
import difflib
from tqdm import tqdm
from bs4 import BeautifulSoup

# --- Scikit-learn imports ---
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================

# 🚨 USE YOUR SERVICE ROLE KEY
HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation"
}

# ==============================================================================
# 1. HELPERS & STEAM SEARCH
# ==============================================================================

def find_steam_id_direct(game_name):
    """Searches Steam directly. No API Key required."""
    print(f"   🔎 Searching Steam for: '{game_name}'...")
    try:
        url = "https://store.steampowered.com/api/storesearch/"
        params = {'term': game_name, 'l': 'english', 'cc': 'US'}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get('total', 0) > 0:
            top = data['items'][0]
            match_name = top['name']
            match_id = int(top['id'])
            
            ratio = difflib.SequenceMatcher(None, game_name.lower(), match_name.lower()).ratio()
            
            if ratio > 0.8:
                print(f"      ✅ Found match: '{match_name}' (ID: {match_id})")
                return match_id
            else:
                print(f"      ❓ Low confidence: '{match_name}' (ID: {match_id})")
                if input(f"      Is this correct? [Y/n]: ").lower() in ['', 'y', 'yes']:
                    return match_id
        
        print(f"      ⚠️ No match found for '{game_name}'")
        return None
    except Exception as e:
        print(f"      ❌ Search Error: {e}")
        return None

def fetch_table(table_name: str) -> pd.DataFrame:
    """Fetches ALL rows with pagination."""
    print(f"Fetching table: {table_name}...")
    all_rows = []
    offset = 0
    while True:
        url = f"{SUPABASE_URL}/rest/v1/{table_name}?select=*&offset={offset}&limit=1000"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if not data: break
            all_rows.extend(data)
            if len(data) < 1000: break
            offset += 1000
        except Exception as e:
            print(f"❌ Error fetching {table_name}: {e}")
            sys.exit(1)
            
    df = pd.DataFrame(all_rows)
    print(f"   -> Loaded {len(df)} rows.")
    return df

def upsert_data(table_name: str, data: list[dict], on_conflict: str = None):
    if not data: return
    print(f"Upserting {len(data)} records into {table_name}...")
    
    url = f"{SUPABASE_URL}/rest/v1/{table_name}"
    if on_conflict:
        url += f"?on_conflict={on_conflict}"
        
    upsert_headers = HEADERS.copy()
    upsert_headers["Prefer"] = "return=representation,resolution=merge-duplicates"
    
    try:
        # Use Pandas for serialization (NaN -> null)
        payload = pd.DataFrame(data).to_json(orient='records', date_format='iso')
        resp = requests.post(url, headers=upsert_headers, data=payload, timeout=30)
        resp.raise_for_status()
        print("   -> Success.")
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error upserting into {table_name}: {e}")
        try: print(f"🔍 Server Response: {e.response.text}")
        except: pass
    except Exception as e:
        print(f"❌ Error upserting into {table_name}: {e}")

# ==============================================================================
# 2. PREDICTION LOGIC
# ==============================================================================

def get_game_details(steam_appid: int) -> dict | None:
    try:
        resp = requests.get(f"https://store.steampowered.com/api/appdetails?appids={steam_appid}&cc=us", timeout=10)
        data = resp.json()
        
        if not data or str(steam_appid) not in data or not data[str(steam_appid)]['success']: 
            return None
            
        g = data[str(steam_appid)]['data']
        price = 0.0
        if not g.get('is_free'):
            price = g.get('price_overview', {}).get('final', 0) / 100.0

        try:
            soup = BeautifulSoup(requests.get(f"https://store.steampowered.com/app/{steam_appid}/", cookies={'birthtime': '568022401'}, timeout=10).text, 'lxml')
            tags = json.dumps([t.get_text(strip=True) for t in soup.find_all('a', class_='app_tag')])
        except: tags = '[]'

        return {
            'appid': steam_appid,
            'name': g.get('name', 'N/A'),
            'developer_genres': ', '.join([x['description'] for x in g.get('genres', [])]),
            'price_usd': price,
            'metacritic_score': g.get('metacritic', {}).get('score', np.nan),
            'release_date': g.get('release_date', {}).get('date', 'N/A'),
            'developers': ', '.join(g.get('developers', ['N/A'])),
            'publishers': ', '.join(g.get('publishers', ['N/A'])),
            'user_tags': tags
        }
    except: return None

# --- ROBUST TAG CLEANER ---
def clean_tags(val):
    """
    Safely parses tags from various formats (JSON list, String repr, etc)
    and formats them for TF-IDF (space separated, underscores for multi-word).
    """
    tags = []
    if isinstance(val, list):
        tags = val
    elif isinstance(val, str):
        try:
            # Try JSON first (for double quotes)
            if val.strip().startswith('['):
                try:
                    tags = json.loads(val)
                except json.JSONDecodeError:
                    # Fallback to AST (for single quotes/python style)
                    tags = ast.literal_eval(val)
            else:
                tags = []
        except:
            tags = []
    
    if not isinstance(tags, list):
        return ''

    # Clean and Join: "Single Player" -> "Single_Player"
    # Then join with space -> "Single_Player Sci_Fi"
    cleaned = [str(t).strip().replace(' ', '_').replace('-', '_') for t in tags]
    return ' '.join(cleaned)

def validate_schema_and_predict(games_df, ratings_df):
    print("\n⚙️  Running Prediction Pipeline...")
    
    merged = pd.merge(ratings_df, games_df, on='game_id', how='left')
    if len(merged) == 0:
        print("❌ MERGE FAILED. No ratings matched with games.")
        return pd.DataFrame(), pd.DataFrame()

    merged['will_skip'] = merged['score'].isnull() | (merged['score'] == 'skipped')
    merged['score_numeric'] = pd.to_numeric(merged['score'], errors='coerce')
    
    # 1. CLEAN TAGS SEPARATELY (No comma replacement here!)
    if 'user_tags' in merged.columns:
        merged['user_tags'] = merged['user_tags'].apply(clean_tags)
    
    # 2. Clean other text columns (Safe to remove commas here)
    for col in ['developer_genres', 'developers', 'publishers']:
        if col in merged.columns:
            merged[col] = merged[col].fillna('').astype(str).str.replace(',', ' ')
    
    merged['metacritic_score'] = pd.to_numeric(merged['metacritic_score'], errors='coerce')
    merged['price_usd'] = pd.to_numeric(merged['price_usd'], errors='coerce')
    merged['release_year'] = pd.to_datetime(merged['release_date'], errors='coerce', format='mixed').dt.year

    num_trans = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    preprocessor = ColumnTransformer(transformers=[
            ('num', num_trans, ['metacritic_score', 'price_usd', 'release_year']),
            ('tags', TfidfVectorizer(max_features=100), 'user_tags')
        ], remainder='drop')

    games_unique = merged.drop_duplicates(subset=['game_id']).copy()
    if len(games_unique) < 2: return pd.DataFrame(), pd.DataFrame()

    preprocessor.fit(games_unique)
    try: feature_names = preprocessor.get_feature_names_out()
    except: feature_names = None

    all_preds = []
    all_importances = []
    critics = merged['critic_id'].unique()

    print(f"   Training models for {len(critics)} critics...")
    for cid in tqdm(critics):
        c_data = merged[merged['critic_id'] == cid]
        if len(c_data) < 5: continue
        
        clf = RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42)
        clf.fit(preprocessor.transform(c_data), c_data['will_skip'])
        
        reg = None
        rated = c_data.dropna(subset=['score_numeric'])
        if len(rated) >= 3:
            reg = RandomForestRegressor(n_estimators=50, random_state=42)
            reg.fit(preprocessor.transform(rated), rated['score_numeric'])

        X_all = preprocessor.transform(games_unique)
        probs = clf.predict_proba(X_all)[:, 1]
        scores = reg.predict(X_all) if reg else np.nan

        preds = games_unique[['game_id', 'name']].copy().rename(columns={'game_id': 'id'})
        preds['critic_id'] = cid
        preds['predicted_skip_probability'] = probs
        preds['predicted_score'] = scores
        all_preds.append(preds)

        if feature_names is not None:
            imp_df = pd.DataFrame({'feature': feature_names, 'importance': clf.feature_importances_})
            imp_df = imp_df.sort_values('importance', ascending=False).head(5)
            imp_df['critic_id'] = cid
            imp_df['model_type'] = 'skip_prediction'
            all_importances.append(imp_df)

            if reg:
                imp_df_reg = pd.DataFrame({'feature': feature_names, 'importance': reg.feature_importances_})
                imp_df_reg = imp_df_reg.sort_values('importance', ascending=False).head(5)
                imp_df_reg['critic_id'] = cid
                imp_df_reg['model_type'] = 'score_prediction'
                all_importances.append(imp_df_reg)

    final_preds = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    final_imps = pd.concat(all_importances, ignore_index=True) if all_importances else pd.DataFrame()
    
    return final_preds, final_imps

# ==============================================================================
# 3. MASTER ORCHESTRATION
# ==============================================================================

def main():
    print("\n🎮 Starting Master Updater (Fixed Tag Parsing) 🎮\n")

    # --- PHASE 1: FILL MISSING DETAILS ---
    games_db = fetch_table('games')
    details_db = fetch_table('games_details')

    if not games_db.empty:
        games_db['id'] = pd.to_numeric(games_db['id'], errors='coerce').astype('Int64')
        
        finished_ids = set()
        if not details_db.empty:
            details_db['id'] = pd.to_numeric(details_db['id'], errors='coerce').astype('Int64')
            valid_mask = details_db['name'].notnull() & (details_db['name'] != '')
            finished_ids = set(details_db[valid_mask]['id'].dropna().unique())

        new_games = games_db[~games_db['id'].isin(finished_ids)]
        
        if not new_games.empty:
            print(f"\n🔎 Processing {len(new_games)} missing/incomplete games...")
            
            new_records = []
            for _, row in new_games.iterrows():
                if not row['game_name']: continue
                
                sid = find_steam_id_direct(row['game_name'])
                if sid:
                    d = get_game_details(sid)
                    if d:
                        d['id'] = int(row['id'])
                        new_records.append(d)
                    else:
                        print("      ⚠️ Failed to get details from Steam.")
                    time.sleep(1)
            
            if new_records:
                upsert_data('games_details', new_records)
        else:
            print("\n✅ All games have details filled.")
    
    # --- PHASE 2: PREDICT ---
    print("\n🔮 Preparing Predictions...")
    details_db = fetch_table('games_details')
    ratings_db = fetch_table('ratings')

    if details_db.empty or ratings_db.empty:
        print("❌ Cannot predict: Tables empty.")
        return

    details_db['game_id'] = pd.to_numeric(details_db['id'], errors='coerce').astype('Int64')
    ratings_db['game_id'] = pd.to_numeric(ratings_db['game_id'], errors='coerce').astype('Int64')
    ratings_db['id'] = pd.to_numeric(ratings_db['id'], errors='coerce').astype('Int64')
    ratings_db = ratings_db.rename(columns={'id': 'rating_id'})

    final_preds, final_imps = validate_schema_and_predict(details_db, ratings_db)

    # --- PHASE 3: UPLOAD ---
    if not final_preds.empty:
        print("   Uploading predictions...")
        # on_conflict='id,critic_id' REQUIRED: Ensure your DB has UNIQUE(id, critic_id)
        upsert_data('critic_predictions', final_preds.to_dict('records'), on_conflict='id,critic_id')

    if not final_imps.empty:
        print("   Uploading feature importances...")
        # on_conflict='critic_id,model_type,feature' REQUIRED: Ensure DB has UNIQUE(critic_id, model_type, feature)
        upsert_data('critic_feature_importances', final_imps.to_dict('records'), on_conflict='critic_id,model_type,feature')

    print("\n✨ Done.")

if __name__ == "__main__":
    main()