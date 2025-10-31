# app.py
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import json
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Election Analytics Dashboard",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .section-header {
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

class DatabaseManager:
    """Handles database setup and management"""
    
    def __init__(self, db_path='election_data.db'):
        self.db_path = db_path
    
    def setup_database(self):
        """Set up the database with cleaned data"""
        st.info("🔄 Setting up election database... This may take a moment.")
        
        try:
            # Read the original CSV
            df = pd.read_csv('All_States_GE.csv', low_memory=False)
            
            st.write(f"📊 Original data shape: {df.shape}")
            
            # Data cleaning steps
            df_clean = df.copy()
            
            # Handle missing values
            df_clean['Sex'] = df_clean['Sex'].fillna('Unknown')
            df_clean['Party'] = df_clean['Party'].fillna('IND')
            df_clean['Candidate_Type'] = df_clean['Candidate_Type'].fillna('GEN')
            df_clean['Deposit_Lost'] = df_clean['Deposit_Lost'].fillna('no')
            
            # Fill numerical missing values
            numerical_cols = ['Votes', 'Turnout_Percentage', 'Vote_Share_Percentage', 'Margin', 'Margin_Percentage']
            for col in numerical_cols:
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].fillna(0)
            
            # Create connection
            conn = sqlite3.connect(self.db_path)
            
            # Save to database
            df_clean.to_sql('election_results', conn, if_exists='replace', index=False)
            
            # Create indexes for performance
            cursor = conn.cursor()
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_year ON election_results(Year)",
                "CREATE INDEX IF NOT EXISTS idx_state ON election_results(State_Name)",
                "CREATE INDEX IF NOT EXISTS idx_party ON election_results(Party)",
                "CREATE INDEX IF NOT EXISTS idx_gender ON election_results(Sex)",
                "CREATE INDEX IF NOT EXISTS idx_position ON election_results(Position)",
                "CREATE INDEX IF NOT EXISTS idx_constituency ON election_results(Constituency_Name)"
            ]
            
            for index_query in indexes:
                cursor.execute(index_query)
            
            conn.commit()
            conn.close()
            
            st.success("✅ Database setup completed successfully!")
            st.write(f"📈 Cleaned data shape: {df_clean.shape}")
            return True
            
        except Exception as e:
            st.error(f"❌ Database setup failed: {str(e)}")
            return False
    
    def check_database_exists(self):
        """Check if database and table exist"""
        if not os.path.exists(self.db_path):
            return False
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='election_results'")
            table_exists = cursor.fetchone() is not None
            
            conn.close()
            return table_exists
        except:
            return False

class ElectionDataAPI:
    """Backend API for serving election data with filters"""
    
    def __init__(self, db_path='election_data.db'):
        self.db_path = db_path
        self.db_manager = DatabaseManager(db_path)
    
    def ensure_database(self):
        """Ensure database is set up before operations"""
        if not self.db_manager.check_database_exists():
            return self.db_manager.setup_database()
        return True
    
    def connect_db(self):
        """Create database connection"""
        return sqlite3.connect(self.db_path)
    
    def get_filtered_data(self, filters=None):
        """
        Get filtered election data based on user selections
        """
        if not self.ensure_database():
            return pd.DataFrame()
            
        if filters is None:
            filters = {}
        
        conn = self.connect_db()
        
        # Base query
        query = """
        SELECT * FROM election_results 
        WHERE 1=1
        """
        params = []
        
        # Apply filters
        if filters.get('year'):
            query += " AND Year = ?"
            params.append(filters['year'])
        
        if filters.get('state') and filters['state'] != 'All States':
            query += " AND State_Name = ?"
            params.append(filters['state'])
        
        if filters.get('party') and filters['party'] != 'All Parties':
            query += " AND Party = ?"
            params.append(filters['party'])
        
        if filters.get('gender') and filters['gender'] != 'All':
            query += " AND Sex = ?"
            params.append(filters['gender'])
        
        if filters.get('constituency') and filters['constituency'] != 'All Constituencies':
            query += " AND Constituency_Name = ?"
            params.append(filters['constituency'])
        
        df = pd.read_sql(query, conn, params=params)
        conn.close()
        return df
    
    def get_available_filters(self):
        """Get available filter options"""
        if not self.ensure_database():
            return {
                'years': [2019],
                'states': ['All States'],
                'parties': ['All Parties'],
                'genders': ['All'],
                'constituencies': ['All Constituencies']
            }
        
        conn = self.connect_db()
        
        filters = {
            'years': [],
            'states': [],
            'parties': [],
            'genders': [],
            'constituencies': []
        }
        
        try:
            # Get unique years
            years_df = pd.read_sql("SELECT DISTINCT Year FROM election_results ORDER BY Year DESC", conn)
            filters['years'] = years_df['Year'].tolist()
            
            # Get unique states
            states_df = pd.read_sql("SELECT DISTINCT State_Name FROM election_results ORDER BY State_Name", conn)
            filters['states'] = ['All States'] + states_df['State_Name'].tolist()
            
            # Get unique parties
            parties_df = pd.read_sql("SELECT DISTINCT Party FROM election_results WHERE Party IS NOT NULL ORDER BY Party", conn)
            filters['parties'] = ['All Parties'] + parties_df['Party'].tolist()
            
            # Get unique genders
            genders_df = pd.read_sql("SELECT DISTINCT Sex FROM election_results WHERE Sex IS NOT NULL ORDER BY Sex", conn)
            filters['genders'] = ['All'] + genders_df['Sex'].tolist()
            
            # Get unique constituencies
            constituencies_df = pd.read_sql("SELECT DISTINCT Constituency_Name FROM election_results ORDER BY Constituency_Name", conn)
            filters['constituencies'] = ['All Constituencies'] + constituencies_df['Constituency_Name'].tolist()
            
        except Exception as e:
            st.error(f"Error loading filter data: {e}")
            
        conn.close()
        return filters
    
    def get_party_seat_share(self, year=None):
        """Get party-wise seat share for a given year"""
        if not self.ensure_database():
            return pd.DataFrame()
            
        conn = self.connect_db()
        
        query = """
        SELECT Party, COUNT(*) as seats_won
        FROM election_results
        WHERE Position = 1
        """
        params = []
        
        if year:
            query += " AND Year = ?"
            params.append(year)
        
        query += " GROUP BY Party ORDER BY seats_won DESC"
        
        df = pd.read_sql(query, conn, params=params)
        conn.close()
        return df
    
    def get_state_turnout(self, year=None):
        """Get state-wise turnout analysis"""
        if not self.ensure_database():
            return pd.DataFrame()
            
        conn = self.connect_db()
        
        query = """
        SELECT State_Name, AVG(Turnout_Percentage) as avg_turnout, 
               MAX(Turnout_Percentage) as max_turnout,
               MIN(Turnout_Percentage) as min_turnout
        FROM election_results
        WHERE Turnout_Percentage IS NOT NULL
        """
        params = []
        
        if year:
            query += " AND Year = ?"
            params.append(year)
        
        query += " GROUP BY State_Name ORDER BY avg_turnout DESC"
        
        df = pd.read_sql(query, conn, params=params)
        conn.close()
        return df
    
    def get_gender_representation(self):
        """Get gender representation over time"""
        if not self.ensure_database():
            return pd.DataFrame()
            
        conn = self.connect_db()
        
        query = """
        SELECT Year, Sex, COUNT(*) as candidate_count,
               SUM(CASE WHEN Position = 1 THEN 1 ELSE 0 END) as winners
        FROM election_results
        WHERE Sex IN ('M', 'F')
        GROUP BY Year, Sex
        ORDER BY Year, Sex
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    
    def get_top_parties_vote_share(self, year=None):
        """Get top parties by vote share"""
        if not self.ensure_database():
            return pd.DataFrame()
            
        conn = self.connect_db()
        
        query = """
        SELECT Party, AVG(Vote_Share_Percentage) as avg_vote_share,
               SUM(Votes) as total_votes
        FROM election_results
        WHERE Vote_Share_Percentage IS NOT NULL
        """
        params = []
        
        if year:
            query += " AND Year = ?"
            params.append(year)
        
        query += " GROUP BY Party HAVING total_votes > 0 ORDER BY avg_vote_share DESC LIMIT 15"
        
        df = pd.read_sql(query, conn, params=params)
        conn.close()
        return df
    
    def get_margin_victory_distribution(self, year=None):
        """Get margin of victory distribution"""
        if not self.ensure_database():
            return pd.DataFrame()
            
        conn = self.connect_db()
        
        query = """
        SELECT Margin_Percentage 
        FROM election_results
        WHERE Position = 1 AND Margin_Percentage IS NOT NULL
        """
        params = []
        
        if year:
            query += " AND Year = ?"
            params.append(year)
        
        df = pd.read_sql(query, conn, params=params)
        conn.close()
        return df
    
    def search_candidates(self, search_term):
        """Search candidates by name or constituency"""
        if not self.ensure_database():
            return pd.DataFrame()
            
        conn = self.connect_db()
        
        query = """
        SELECT Candidate, Party, Constituency_Name, State_Name, Year, 
               Votes, Vote_Share_Percentage, Position
        FROM election_results
        WHERE Candidate LIKE ? OR Constituency_Name LIKE ?
        ORDER BY Year DESC, Votes DESC
        LIMIT 100
        """
        
        search_pattern = f'%{search_term}%'
        df = pd.read_sql(query, conn, params=(search_pattern, search_pattern))
        conn.close()
        return df

# Initialize API
api = ElectionDataAPI()

# Main Dashboard
def main():
    # Header
    st.markdown('<h1 class="main-header">🗳️ Election Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Check if data file exists
    if not os.path.exists('All_States_GE.csv'):
        st.error("❌ Data file 'All_States_GE.csv' not found. Please ensure it's in the same directory.")
        st.stop()
    
    # Initialize database if needed
    if not api.ensure_database():
        st.error("❌ Failed to initialize database. Please check the data file.")
        st.stop()
    
    # Sidebar for filters
    st.sidebar.header("🔍 Filter Data")
    
    # Get available filters
    filters_data = api.get_available_filters()
    
    # Filter widgets
    selected_year = st.sidebar.selectbox("Select Year", filters_data['years'])
    selected_state = st.sidebar.selectbox("Select State", filters_data['states'])
    selected_party = st.sidebar.selectbox("Select Party", filters_data['parties'])
    selected_gender = st.sidebar.selectbox("Select Gender", filters_data['genders'])
    selected_constituency = st.sidebar.selectbox("Select Constituency", filters_data['constituencies'])
    
    # Apply filters
    current_filters = {
        'year': selected_year,
        'state': selected_state,
        'party': selected_party,
        'gender': selected_gender,
        'constituency': selected_constituency
    }
    
    # Get filtered data
    filtered_data = api.get_filtered_data(current_filters)
    
    # Key Metrics
    st.markdown("### 📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_candidates = len(filtered_data)
        st.metric("Total Candidates", f"{total_candidates:,}")
    
    with col2:
        winners = len(filtered_data[filtered_data['Position'] == 1])
        st.metric("Winners", f"{winners:,}")
    
    with col3:
        if len(filtered_data) > 0:
            avg_turnout = filtered_data['Turnout_Percentage'].mean()
            st.metric("Avg Turnout", f"{avg_turnout:.1f}%")
        else:
            st.metric("Avg Turnout", "N/A")
    
    with col4:
        female_candidates = len(filtered_data[filtered_data['Sex'] == 'F'])
        st.metric("Female Candidates", f"{female_candidates:,}")

    # Visualizations Section
    st.markdown("---")
    
    # Row 1: Party-wise Seat Share and State-wise Turnout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 class="section-header">🏛️ Party-wise Seat Share</h3>', unsafe_allow_html=True)
        seat_share_data = api.get_party_seat_share(selected_year)
        
        if not seat_share_data.empty:
            fig = px.bar(
                seat_share_data.head(10),
                x='Party',
                y='seats_won',
                title=f"Top 10 Parties by Seats Won ({selected_year})",
                color='seats_won',
                color_continuous_scale='blues'
            )
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for the selected filters")
    
    with col2:
        st.markdown('<h3 class="section-header">🗺️ State-wise Turnout Analysis</h3>', unsafe_allow_html=True)
        turnout_data = api.get_state_turnout(selected_year)
        
        if not turnout_data.empty:
            # Create a bar chart instead of map for simplicity
            fig = px.bar(
                turnout_data.head(15),
                x='State_Name',
                y='avg_turnout',
                title=f"Average Voter Turnout by State ({selected_year})",
                color='avg_turnout',
                color_continuous_scale='viridis'
            )
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No turnout data available")

    # Row 2: Gender Representation and Top Parties by Vote Share
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 class="section-header">👥 Gender Representation Over Time</h3>', unsafe_allow_html=True)
        gender_data = api.get_gender_representation()
        
        if not gender_data.empty:
            # Pivot for better visualization
            gender_pivot = gender_data.pivot(index='Year', columns='Sex', values='candidate_count').fillna(0)
            
            fig = px.line(
                gender_pivot.reset_index(),
                x='Year',
                y=['M', 'F'],
                title="Gender Representation Over Time",
                labels={'value': 'Number of Candidates', 'variable': 'Gender'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No gender data available")
    
    with col2:
        st.markdown('<h3 class="section-header">📈 Top Parties by Vote Share</h3>', unsafe_allow_html=True)
        vote_share_data = api.get_top_parties_vote_share(selected_year)
        
        if not vote_share_data.empty:
            fig = px.pie(
                vote_share_data.head(8),
                values='avg_vote_share',
                names='Party',
                title=f"Top Parties by Average Vote Share ({selected_year})",
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No vote share data available")

    # Row 3: Margin of Victory and Search
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 class="section-header">📊 Margin of Victory Distribution</h3>', unsafe_allow_html=True)
        margin_data = api.get_margin_victory_distribution(selected_year)
        
        if not margin_data.empty:
            fig = px.histogram(
                margin_data,
                x='Margin_Percentage',
                nbins=20,
                title=f"Distribution of Victory Margins ({selected_year})",
                labels={'Margin_Percentage': 'Margin of Victory (%)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No margin data available")
    
    with col2:
        st.markdown('<h3 class="section-header">🔍 Search Candidates/Constituencies</h3>', unsafe_allow_html=True)
        search_term = st.text_input("Enter candidate name or constituency:")
        
        if search_term:
            search_results = api.search_candidates(search_term)
            
            if not search_results.empty:
                st.dataframe(
                    search_results,
                    use_container_width=True,
                    height=400
                )
            else:
                st.warning("No results found for your search term")

    # Analytical Scenarios Section
    st.markdown("---")
    st.markdown('<h2 class="section-header">🔬 Analytical Scenarios</h2>', unsafe_allow_html=True)
    
    # Scenario 1: Highest voter turnout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**a. State with Highest Voter Turnout**")
        latest_year = max(filters_data['years'])
        turnout_data = api.get_state_turnout(latest_year)
        
        if not turnout_data.empty:
            top_state = turnout_data.iloc[0]
            st.success(f"🏆 {top_state['State_Name']} had the highest average turnout of {top_state['avg_turnout']:.1f}% in {latest_year}")
            
            # Show top 5 states
            st.write("Top 5 States by Turnout:")
            st.dataframe(turnout_data.head()[['State_Name', 'avg_turnout']])
    
    with col2:
        st.markdown("**b. Party Seat Changes Between Elections**")
        if len(filters_data['years']) >= 2:
            recent_years = sorted(filters_data['years'], reverse=True)[:2]
            
            seats_year1 = api.get_party_seat_share(recent_years[0])
            seats_year2 = api.get_party_seat_share(recent_years[1])
            
            # Merge to compare
            comparison = pd.merge(seats_year1, seats_year2, on='Party', how='outer', suffixes=('_current', '_previous'))
            comparison = comparison.fillna(0)
            comparison['seat_change'] = comparison['seats_won_current'] - comparison['seats_won_previous']
            
            max_gain = comparison.loc[comparison['seat_change'].idxmax()]
            max_loss = comparison.loc[comparison['seat_change'].idxmin()]
            
            st.info(f"📈 **Biggest Gain**: {max_gain['Party']} (+{int(max_gain['seat_change'])} seats)")
            st.error(f"📉 **Biggest Loss**: {max_loss['Party']} ({int(max_loss['seat_change'])} seats)")
    
    # Scenario 3: Women candidates percentage
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**c. Women Candidates Percentage**")
        gender_stats = api.get_gender_representation()
        
        if not gender_stats.empty:
            total_candidates = gender_stats['candidate_count'].sum()
            women_candidates = gender_stats[gender_stats['Sex'] == 'F']['candidate_count'].sum()
            women_percentage = (women_candidates / total_candidates) * 100
            
            st.metric("Women Candidates Overall", f"{women_percentage:.1f}%")
            
            # Show trend
            women_trend = gender_stats[gender_stats['Sex'] == 'F'].set_index('Year')['candidate_count']
            st.line_chart(women_trend)
    
    with col2:
        st.markdown("**d. Narrowest Victory Margins**")
        margin_data = api.get_margin_victory_distribution()
        
        if not margin_data.empty:
            # Get constituencies with smallest margins
            conn = api.connect_db()
            narrow_wins = pd.read_sql("""
                SELECT Constituency_Name, State_Name, Margin_Percentage, Candidate, Party
                FROM election_results
                WHERE Position = 1 AND Margin_Percentage IS NOT NULL
                ORDER BY Margin_Percentage ASC
                LIMIT 10
            """, conn)
            conn.close()
            
            st.dataframe(narrow_wins[['Constituency_Name', 'State_Name', 'Margin_Percentage', 'Candidate', 'Party']])

    # Scenario 5: National vs Regional parties over time
    st.markdown("**e. National vs Regional Parties Vote Share Over Time**")
    
    # Simple classification (you can enhance this)
    national_parties = ['BJP', 'INC', 'CPI', 'CPM', 'BSP']
    
    conn = api.connect_db()
    party_trends = pd.read_sql("""
        SELECT Year, Party, AVG(Vote_Share_Percentage) as avg_vote_share
        FROM election_results
        WHERE Vote_Share_Percentage IS NOT NULL
        GROUP BY Year, Party
    """, conn)
    conn.close()
    
    party_trends['Party_Type'] = party_trends['Party'].apply(
        lambda x: 'National' if x in national_parties else 'Regional'
    )
    
    type_trends = party_trends.groupby(['Year', 'Party_Type'])['avg_vote_share'].mean().reset_index()
    
    if not type_trends.empty:
        fig = px.line(
            type_trends,
            x='Year',
            y='avg_vote_share',
            color='Party_Type',
            title="National vs Regional Parties Vote Share Over Time"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Election Analytics Dashboard • Data Source: All_States_GE.csv • Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()