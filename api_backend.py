# api_backend.py
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import sqlite3
import uvicorn
from typing import Optional, List
import json

# Initialize FastAPI app
app = FastAPI(
    title="Election Data API",
    description="A comprehensive API for Indian election data analysis with advanced filtering",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc"  # Alternative documentation
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ElectionDataAPI:
    def __init__(self, db_path='election_data.db'):
        self.db_path = db_path
    
    def connect_db(self):
        return sqlite3.connect(self.db_path)
    
    def get_filtered_data(self, year: Optional[int] = None, 
                         state: Optional[str] = None,
                         party: Optional[str] = None,
                         gender: Optional[str] = None,
                         constituency: Optional[str] = None,
                         position: Optional[int] = None,
                         min_votes: Optional[int] = None,
                         min_vote_share: Optional[float] = None,
                         limit: int = 100):
        conn = self.connect_db()
        
        query = "SELECT * FROM election_results WHERE 1=1"
        params = []
        
        if year:
            query += " AND Year = ?"
            params.append(year)
        
        if state:
            query += " AND State_Name = ?"
            params.append(state)
        
        if party:
            query += " AND Party = ?"
            params.append(party)
        
        if gender:
            query += " AND Sex = ?"
            params.append(gender)
        
        if constituency:
            query += " AND Constituency_Name = ?"
            params.append(constituency)
        
        if position:
            query += " AND Position = ?"
            params.append(position)
        
        if min_votes:
            query += " AND Votes >= ?"
            params.append(min_votes)
        
        if min_vote_share:
            query += " AND Vote_Share_Percentage >= ?"
            params.append(min_vote_share)
        
        query += f" LIMIT {limit}"
        
        df = pd.read_sql(query, conn, params=params)
        conn.close()
        return df
    
    def get_available_filters(self):
        """Get available options for each filter"""
        conn = self.connect_db()
        
        filters = {}
        
        # Get available years
        years_df = pd.read_sql("SELECT DISTINCT Year FROM election_results ORDER BY Year DESC", conn)
        filters['years'] = years_df['Year'].tolist()
        
        # Get available states
        states_df = pd.read_sql("SELECT DISTINCT State_Name FROM election_results ORDER BY State_Name", conn)
        filters['states'] = states_df['State_Name'].tolist()
        
        # Get available parties
        parties_df = pd.read_sql("SELECT DISTINCT Party FROM election_results WHERE Party IS NOT NULL ORDER BY Party", conn)
        filters['parties'] = parties_df['Party'].tolist()
        
        # Get available genders
        genders_df = pd.read_sql("SELECT DISTINCT Sex FROM election_results WHERE Sex IS NOT NULL ORDER BY Sex", conn)
        filters['genders'] = genders_df['Sex'].tolist()
        
        # Get available constituencies
        constituencies_df = pd.read_sql("SELECT DISTINCT Constituency_Name FROM election_results ORDER BY Constituency_Name", conn)
        filters['constituencies'] = constituencies_df['Constituency_Name'].tolist()
        
        conn.close()
        return filters

# Initialize API
election_api = ElectionDataAPI()

# Response Models
class CandidateResponse(BaseModel):
    Candidate: str
    Party: str
    Constituency_Name: str
    State_Name: str
    Year: int
    Votes: int
    Vote_Share_Percentage: float
    Position: int
    Sex: str
    Deposit_Lost: str

class PartyPerformance(BaseModel):
    Party: str
    seats_won: int
    total_votes: int
    avg_vote_share: float

class StateTurnout(BaseModel):
    State_Name: str
    avg_turnout: float
    max_turnout: float
    min_turnout: float

class FilterOptions(BaseModel):
    years: List[int]
    states: List[str]
    parties: List[str]
    genders: List[str]
    constituencies: List[str]

# API Routes
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Election Data API",
        "version": "1.0.0",
        "description": "Comprehensive Indian election data with advanced filtering",
        "filters_available": {
            "year": "Filter by election year",
            "state": "Filter by state/UT", 
            "party": "Filter by political party",
            "gender": "Filter by candidate gender (M/F)",
            "constituency": "Filter by constituency name",
            "position": "Filter by position/rank (1=winner)",
            "min_votes": "Filter by minimum votes",
            "min_vote_share": "Filter by minimum vote share %"
        },
        "endpoints": {
            "docs": "/docs",
            "filters": "/api/filters",
            "candidates": "/api/candidates",
            "party-performance": "/api/party-performance",
            "state-turnout": "/api/state-turnout",
            "gender-stats": "/api/gender-stats",
            "search": "/api/search",
            "analytics": "/api/analytics/highest-turnout"
        }
    }

@app.get("/api/filters", response_model=FilterOptions)
async def get_available_filters():
    """
    Get all available filter options
    """
    try:
        filters = election_api.get_available_filters()
        return filters
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/candidates", response_model=List[CandidateResponse])
async def get_candidates(
    year: Optional[int] = Query(None, description="Filter by election year"),
    state: Optional[str] = Query(None, description="Filter by state"),
    party: Optional[str] = Query(None, description="Filter by party"),
    gender: Optional[str] = Query(None, description="Filter by gender (M/F)"),
    constituency: Optional[str] = Query(None, description="Filter by constituency"),
    position: Optional[int] = Query(None, description="Filter by position (1=winner)"),
    min_votes: Optional[int] = Query(None, description="Minimum votes received"),
    min_vote_share: Optional[float] = Query(None, description="Minimum vote share percentage"),
    limit: int = Query(100, description="Number of records to return")
):
    """
    Get election candidates with advanced filtering options
    
    **Examples:**
    - `/api/candidates?year=2019&state=Delhi` - 2019 Delhi candidates
    - `/api/candidates?party=BJP&position=1` - BJP winners
    - `/api/candidates?gender=F&min_vote_share=10` - Female candidates with >10% vote share
    - `/api/candidates?constituency=Varanasi` - Varanasi constituency candidates
    """
    try:
        data = election_api.get_filtered_data(
            year=year, state=state, party=party, 
            gender=gender, constituency=constituency,
            position=position, min_votes=min_votes,
            min_vote_share=min_vote_share, limit=limit
        )
        return data.to_dict('records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/party-performance", response_model=List[PartyPerformance])
async def get_party_performance(
    year: Optional[int] = Query(None, description="Filter by year"),
    state: Optional[str] = Query(None, description="Filter by state")
):
    """
    Get party performance statistics with state filtering
    """
    try:
        conn = election_api.connect_db()
        
        query = """
        SELECT Party, 
               COUNT(*) as seats_won,
               SUM(Votes) as total_votes,
               AVG(Vote_Share_Percentage) as avg_vote_share
        FROM election_results
        WHERE Position = 1
        """
        params = []
        
        if year:
            query += " AND Year = ?"
            params.append(year)
        
        if state:
            query += " AND State_Name = ?"
            params.append(state)
        
        query += " GROUP BY Party ORDER BY seats_won DESC"
        
        df = pd.read_sql(query, conn, params=params)
        conn.close()
        return df.to_dict('records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/state-turnout", response_model=List[StateTurnout])
async def get_state_turnout(
    year: Optional[int] = Query(None, description="Filter by year"),
    min_turnout: Optional[float] = Query(None, description="Minimum average turnout")
):
    """
    Get state-wise voter turnout statistics
    """
    try:
        conn = election_api.connect_db()
        
        query = """
        SELECT State_Name, 
               AVG(Turnout_Percentage) as avg_turnout,
               MAX(Turnout_Percentage) as max_turnout,
               MIN(Turnout_Percentage) as min_turnout
        FROM election_results
        WHERE Turnout_Percentage IS NOT NULL
        """
        params = []
        
        if year:
            query += " AND Year = ?"
            params.append(year)
        
        query += " GROUP BY State_Name"
        
        if min_turnout:
            query += " HAVING avg_turnout >= ?"
            params.append(min_turnout)
        
        query += " ORDER BY avg_turnout DESC"
        
        df = pd.read_sql(query, conn, params=params)
        conn.close()
        return df.to_dict('records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/gender-stats")
async def get_gender_statistics(
    year: Optional[int] = Query(None, description="Filter by year"),
    state: Optional[str] = Query(None, description="Filter by state")
):
    """
    Get gender representation statistics with filtering
    """
    try:
        conn = election_api.connect_db()
        
        query = """
        SELECT Year, Sex, 
               COUNT(*) as candidate_count,
               SUM(CASE WHEN Position = 1 THEN 1 ELSE 0 END) as winners
        FROM election_results
        WHERE Sex IN ('M', 'F')
        """
        params = []
        
        if year:
            query += " AND Year = ?"
            params.append(year)
        
        if state:
            query += " AND State_Name = ?"
            params.append(state)
        
        query += " GROUP BY Year, Sex ORDER BY Year, Sex"
        
        df = pd.read_sql(query, conn, params=params)
        conn.close()
        
        # Calculate percentages
        result = []
        for year_val in df['Year'].unique():
            year_data = df[df['Year'] == year_val]
            total = year_data['candidate_count'].sum()
            
            for _, row in year_data.iterrows():
                result.append({
                    "year": int(row['Year']),
                    "gender": row['Sex'],
                    "candidate_count": int(row['candidate_count']),
                    "winners": int(row['winners']),
                    "percentage": round((row['candidate_count'] / total) * 100, 2) if total > 0 else 0
                })
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/search")
async def search_candidates(
    q: str = Query(..., description="Search term for candidate or constituency"),
    year: Optional[int] = Query(None, description="Filter by year"),
    state: Optional[str] = Query(None, description="Filter by state")
):
    """
    Search candidates by name or constituency with additional filters
    """
    try:
        conn = election_api.connect_db()
        
        query = """
        SELECT Candidate, Party, Constituency_Name, State_Name, Year, 
               Votes, Vote_Share_Percentage, Position, Sex
        FROM election_results
        WHERE (Candidate LIKE ? OR Constituency_Name LIKE ?)
        """
        params = ['%' + q + '%', '%' + q + '%']
        
        if year:
            query += " AND Year = ?"
            params.append(year)
        
        if state:
            query += " AND State_Name = ?"
            params.append(state)
        
        query += " ORDER BY Year DESC, Votes DESC LIMIT 50"
        
        df = pd.read_sql(query, conn, params=params)
        conn.close()
        return df.to_dict('records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/highest-turnout")
async def get_highest_turnout(
    year: Optional[int] = Query(None, description="Specific year to analyze")
):
    """
    Get state with highest voter turnout
    """
    try:
        conn = election_api.connect_db()
        
        if not year:
            # Get latest year if not specified
            year_df = pd.read_sql("SELECT MAX(Year) as latest_year FROM election_results", conn)
            year = year_df.iloc[0]['latest_year']
        
        # Get state with highest turnout for the year
        turnout_df = pd.read_sql("""
            SELECT State_Name, AVG(Turnout_Percentage) as avg_turnout
            FROM election_results
            WHERE Year = ? AND Turnout_Percentage IS NOT NULL
            GROUP BY State_Name
            ORDER BY avg_turnout DESC
            LIMIT 1
        """, conn, params=[year])
        
        conn.close()
        
        if not turnout_df.empty:
            return {
                "year": year,
                "state": turnout_df.iloc[0]['State_Name'],
                "average_turnout": round(turnout_df.iloc[0]['avg_turnout'], 2),
                "analysis": f"Highest voter turnout state in {year}"
            }
        else:
            raise HTTPException(status_code=404, detail=f"No turnout data found for year {year}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/close-contests")
async def get_close_contests(
    year: Optional[int] = Query(None, description="Filter by year"),
    max_margin: float = Query(5.0, description="Maximum margin percentage")
):
    """
    Get constituencies with closest victory margins
    """
    try:
        conn = election_api.connect_db()
        
        query = """
        SELECT Constituency_Name, State_Name, Candidate, Party, 
               Margin_Percentage, Votes, Vote_Share_Percentage
        FROM election_results
        WHERE Position = 1 AND Margin_Percentage IS NOT NULL
        AND Margin_Percentage <= ?
        """
        params = [max_margin]
        
        if year:
            query += " AND Year = ?"
            params.append(year)
        
        query += " ORDER BY Margin_Percentage ASC LIMIT 20"
        
        df = pd.read_sql(query, conn, params=params)
        conn.close()
        return df.to_dict('records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)