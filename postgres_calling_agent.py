import os
import streamlit as st
from typing import Any, List, Optional, Mapping
import google.generativeai as genai
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import BaseLLM
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from dotenv import load_dotenv

class GeminiLLMWrapper(BaseLLM):
    """Wrapper class to make Gemini compatible with LangChain's expectations"""
    
    def __init__(self, api_key: str):
        super().__init__()
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel("gemini-1.5-flash")
    
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        generations = []
        for prompt in prompts:
            response = self._model.generate_content(prompt)
            generations.append([{"text": response.text, "generation_info": {}}])
        return {"generations": generations}

    @property
    def _llm_type(self) -> str:
        return "gemini"

def create_database_connection() -> SQLDatabase:
    """Create database connection using environment variables"""
    db_config = {
        "dbname": st.secrets["LOCAL_DBNAME"],
        "password": st.secrets["LOCAL_DBPASS"],
        "user": st.secrets["LOCAL_DBUSER"],
        "host": st.secrets["LOCAL_DBHOST"],
        "port": st.secrets["LOCAL_DBPORT"]
    }
    
    # Validate configuration
    missing_vars = [k for k, v in db_config.items() if not v]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    # Construct database URL
    db_url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
    
    try:
        return SQLDatabase.from_uri(db_url)
    except Exception as e:
        raise RuntimeError(f"Database connection failed: {e}")

def setup_sql_agent():
    """Initialize and configure the SQL agent"""
    try:
        # Initialize components
        llm = GeminiLLMWrapper(api_key=st.secrets["GOOGLE_API_KEY"])
        db = create_database_connection()
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)

        # Create agent with specific configuration
        agent_executor = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True
        )
        
        return agent_executor
    except Exception as e:
        raise RuntimeError(f"Failed to set up SQL agent: {e}")

def main():
    st.set_page_config(page_title="SQL Query Assistant", layout="wide")
    
    st.title("SQL Query Assistant with Google Gemini")
    st.write("Ask questions about your database in natural language!")

    # Display database schema if connection is successful
    if 'agent' not in st.session_state:
        try:
            st.session_state.agent = setup_sql_agent()
            db = create_database_connection()
            st.success("SQL Agent initialized successfully!")
            
            # Show database schema in an expander
            with st.expander("View Database Schema"):
                st.code(db.get_table_info())
                
        except Exception as e:
            st.error(f"Failed to initialize agent: {str(e)}")
            return

    # Query input
    query = st.text_area("Enter your question about the database:", height=100)
    
    if st.button("Run Query"):
        if query:
            try:
                with st.spinner("Processing your query..."):
                    response = st.session_state.agent.run(query)
                
                # Display response in a nice format
                st.subheader("Response:")
                st.write(response)
                
                # Add to history
                if 'history' not in st.session_state:
                    st.session_state.history = []
                st.session_state.history.append({"query": query, "response": response})
                
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                st.error("Full error message:")
                st.code(str(e))
        else:
            st.warning("Please enter a query first.")
    
    # Display history
    if st.session_state.get('history'):
        st.subheader("Query History")
        for i, item in enumerate(reversed(st.session_state.history), 1):
            with st.expander(f"Query {i}: {item['query'][:50]}..."):
                st.text("Query:")
                st.code(item['query'])
                st.text("Response:")
                st.write(item['response'])

if __name__ == "__main__":
    main()