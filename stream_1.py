import streamlit as st
import pandas as pd
import sqlalchemy
# test
from sqlalchemy import create_engine, text
from typing import Dict, Any
from groq import Groq
import matplotlib.pyplot as plt
import seaborn as sns 

api_key = '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'
Client = Groq(api_key=api_key)

# Hide the GitHub icon/link
st.set_page_config(
    page_title="Your App Title",
    page_icon="🧊",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Read glossary
with open('rag.txt', 'r') as f:
    glossary = f.read()

class FlexibleQuerySystem:
    def __init__(self, glossary: str):
        self.engine = None
        self.df = None
        self.table_info = None
        self.glossary = glossary

    def load_data(self, csv_file: str, db_name: str = "serviceai.db", table_name: str = "serviceai"):
        """Loads data from a CSV file and sets up the SQLite engine."""
        self.df = pd.read_csv(csv_file)
        self.prepare_data(db_name, table_name)
        self.table_info = self.get_table_info(table_name)
    
    def prepare_data(self, db_name: str, table_name: str):
        """Prepares the data and stores it in an SQLite database."""
        self.df.columns = [col.lower().replace(' ', '_') for col in self.df.columns]
        self.engine = create_engine(f'sqlite:///{db_name}')
        self.df.to_sql(table_name, self.engine, if_exists='replace', index=False)
        
    def get_table_info(self, table_name: str) -> str:
        """Generates and returns the table schema information."""
        columns = self.df.columns.tolist()
        dtypes = self.df.dtypes.tolist()
        table_info = f"Table name: {table_name}\n"
        table_info += "Columns:\n"
        for col, dtype in zip(columns, dtypes):
            table_info += f"- {col} : ({dtype})\n"
        return table_info
    
    def clean_sql_query(self, sql_query: str) -> str:
        """Cleans up the SQL query by removing unnecessary backticks and other unwanted characters."""
        return sql_query.replace('```', '').strip()

    def natural_language_to_sql(self, query: str, previous_context: list) -> str:
        """Converts a natural language query into an SQL query and returns the SQL query and token usage."""
        prompt = f"""
        Respond with only the SQL query, nothing else.
        
        Given Table information:
        {self.table_info}
        
        Knowledge Base:
        {self.glossary}
             
        Convert this natural language query to SQL:
        {query}
        
        Extra context about columns:
service_id:
    Definition: Unique identifier for each maintenance/service record
    Usage: Primary key, never duplicated

equipment_id:
    Definition: Identifier for specific equipment/machine
    Usage: Same equipment may have multiple maintenance records
    Note: Track maintenance history per equipment

service_task:
    Definition: Type of task - either 'Maintenance' or 'Service'
    Usage: Filter by task type using case-insensitive comparison
    Note: Services are scheduled checks, Maintenance includes specific activities

maintenance_frequency:
    Definition: Hours between required maintenance activities
    Usage: Common values: 5, 100, 250, 500, 1500 hours
    Note: Different frequencies for different maintenance types

last_serviced_hmr:
    Definition: Hour meter reading when service was performed
    Usage: Track actual service timing
    Note: Compare with next_due_hmr for compliance

next_due_hmr:
    Definition: Hour meter reading when next service is due
    Usage: Calculate when maintenance is due
    Note: next_due_hmr - last_serviced_hmr should equal maintenance_frequency

frequency_unit:
    Definition: Unit of measurement for service intervals
    Usage: Currently always "hours"
    Note: Important for standardizing time calculations

service_activity:
    Definition: Specific maintenance action performed
    Values: Filter Change, Oil Change, Greasing, Coolant Top up, etc.
    Usage: Track maintenance types and patterns

make:
    Definition: Equipment manufacturer
    Usage: Group analysis by manufacturer
    Examples: JCB, CATERPILLAR, HYUNDAI, TATA PRIME, ACE, ASHOK LEYLAND, ATLAST COPCO, FIAT, TATA HITACHI, TOYOTA, VOLVO.

model:
    Definition: Equipment model number/name
    Usage: Specific model identification
    Examples: 415, 2830, 1150H-LGP, 15L-7M, 2518T, 336D2L, 3DX, 430ZX PLUS, 432ZXC3, 4DX, 90E, 950H, AF50D, CAT320, D6R2, etc.
    Note: Analyze maintenance patterns by model

machine_type:
    Definition: Category of heavy equipment
    Usage: Group equipment by type
    Examples: Excavator, Backhoe Loader, Dozer, etc.
        
        
        These are the questions asked by the user and the responses provided by the system. This is the context for your query generation:
        {previous_context}
        
        YOU ARE AN SQL EXPERT. ONLY RESPOND WITH SQL QUERY.
        SQL Query:
        """
        
        response = Client.chat.completions.create(
            messages = [
                {
                    "role" : "system",
                    "content" : "You are an SQL expert. Convert the natural queries into SQL using the table information provided."
                },
                {
                    "role" : "user",
                    "content" : prompt
                }
            ],
            model = "llama3-70b-8192",
        )
        
        
        sql_query = response.choices[0].message.content.strip()
        tokens_used = response.usage.total_tokens
        
        return self.clean_sql_query(sql_query), tokens_used
    
    def analyze_results(self, query_result: pd.DataFrame, user_query: str) -> str:
        """Analyzes the query results and provides insights using LLM."""
        # Convert DataFrame to a readable format
        result_description = query_result.to_string()
        
        analysis_prompt = f"""
        Analyze the following query results and provide insights. Be concise but comprehensive.
        
        Original User Query: {user_query}
        
        Query Results:
        {result_description}

        Table Context:
        {self.table_info}

        Please provide:
        1. A summary of the key findings
        2. Any notable patterns or trends
        3. Business-relevant insights
        4. Potential areas for further investigation
        
        Restrict the response to 3-4 sentences. Use Bullet points for clarity.
        Your response is shown along with query results and should seem like an AI is providing insights. 

        Keep the response clear, concise and actionable.
        """

        response = Client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a data analyst expert providing insights from query results."
                },
                {
                    "role": "user",
                    "content": analysis_prompt
                }
            ],
            model="llama3-70b-8192",
        )

        return response.choices[0].message.content.strip()
    
    def execute_query(self, sql_query: str) -> pd.DataFrame:
        """Executes the SQL query and returns the result as a DataFrame."""
        with self.engine.connect() as conn:
            return pd.read_sql(text(sql_query), conn)
        
    def process_query(self, user_query: str, previous_context: list) -> Dict[str, Any]:
        """Processes the user's natural language query."""
        sql_query, tokens_used = self.natural_language_to_sql(user_query, previous_context)
        result_df = self.execute_query(sql_query)
        
        analysis = self.analyze_results(result_df, user_query)
        
        return {
            'user_query': user_query,
            'sql_query': sql_query,
            'tokens_used': tokens_used,
            'result': result_df.to_dict(orient='records'),
            'analysis': analysis
        }

    def update_glossary(self, feedback: str):
        """Updates the glossary based on user feedback."""
        self.glossary += f"\n{feedback}"
    
    def reset(self):
        """Resets the session state."""
        self.previous_context = []
        self.glossary = glossary
    
    def plot_data(self, df: pd.DataFrame):
        """Generates a plot for the given DataFrame."""
        st.subheader("Plot of Query Results")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=df.columns[0], y=df.columns[1], data=df, ax=ax)
        ax.set_xlabel(df.columns[0])
        ax.set_ylabel(df.columns[1])
        ax.set_title("Plot of Query Results")
        st.pyplot(fig)
    
# Streamlit interface

def main():
    st.title("Flexible Query System with SQL Generation")

    # Initialize session state for context and token count if not already present
    if 'context' not in st.session_state:
        st.session_state.context = []
    if 'total_token' not in st.session_state:
        st.session_state.total_token = 0

    # Reset button
    if st.button("Reset"):
        # query_system.reset()
        st.session_state.context = []
        st.session_state.total_token = 0
        st.success("Session has been reset.")

    query_system = FlexibleQuerySystem(glossary)

    # csv_file = 'df1.csv'  # Replace with your file path
    csv_file = 'Service Insights by AI - YL.csv'
    db_file = 'serviceai.db'
    table_name = 'serviceai'

    # Load data and display the table schema
    query_system.load_data(csv_file, db_file, table_name)
    
    # Input for user query
    user_query = st.text_input("Enter your query:")
    
    if user_query:
        try:
            # Process the query
            result = query_system.process_query(user_query, st.session_state.context)
            
            # Update the session state with new context and token usage
            st.session_state.context.append({"role": "user", "content": user_query})
            st.session_state.context.append({"role": "assistant", "content": result['sql_query']})
            st.session_state.total_token += result['tokens_used']
            
            # Display the generated SQL query
            # st.subheader("Generated SQL Query")
            # st.code(result['sql_query'], language='sql')
            
            # # Display tokens used
            # st.subheader("Tokens Used")
            # st.write(result['tokens_used'])
            
            # # Display the total token count for the session
            # st.subheader("Total Tokens Used")
            # st.write(st.session_state.total_token)
            
            # Display the result as a DataFrame
            st.subheader("Query Results")
            st.dataframe(pd.DataFrame(result['result']))
            
            st.subheader("Analysis")
            st.write(result['analysis'])
            
            df_plot = pd.DataFrame(result['result'])
            
            if st.button("Show Plot"):
                query_system.plot_data(df_plot)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
