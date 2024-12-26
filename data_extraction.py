import re
import os
import csv
import psycopg2
from dotenv import load_dotenv
from num2words import num2words
import logging
from transformers import pipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

load_dotenv()

DBNAME = os.environ.get("LOCAL_DBNAME", "")
DBPASS = os.environ.get("LOCAL_DBPASS", "")
DBUSER = os.environ.get("LOCAL_DBUSER", "")
DBHOST = os.environ.get("LOCAL_DBHOST", "")
DBPORT = os.environ.get("LOCAL_DBPORT", "")

# List of SQL reserved keywords
SQL_KEYWORDS = [
    "select", "insert", "update", "delete", "from", "where", "join", "group", "order", "by", "having", "limit", 
    "count", "distinct", "drop", "alter", "create", "table", "index", "database", "view", "primary", "foreign", 
    "key", "null", "and", "or", "not", "as", "like", "in", "between", "exists", "union", "intersect", "except", 
    "case", "when", "then", "else", "end", "default", "check", "unique", "cascade"
]

def shorten_column_name(name, max_length=63):
    """
    Shorten the column name by truncating the middle part and replacing it with 'xxx'
    if the length exceeds the max_length.

    Parameters:
        name (str): The original column name.
        max_length (int): The maximum allowed length of the name.

    Returns:
        str: The shortened column name.
    """
    if len(name) <= max_length:
        return name

    # Calculate the length of the prefix and suffix to retain
    prefix_length = (max_length - 3) // 2
    suffix_length = max_length - 3 - prefix_length

    # Form the new shortened name
    shortened_name = name[:prefix_length] + 'xxx' + name[-suffix_length:]
    return shortened_name

def connect_to_db(db_name, user, password, host, port):
    """Establish a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            dbname=db_name,
            user=user,
            password=password,
            host=host,
            port=port
        )
        logger.info("Connected to the database.")
        return conn
    except Exception as e:
        logger.error("Error connecting to the database: %s", e)
        raise

def format_column_name(name):
    """Convert column name to lowercase, replace spaces with underscores, remove special characters, 
    convert numbers to words, replace % with 'percent', and replace SQL keywords."""
    # Replace spaces with underscores, convert to lowercase, and remove unwanted characters
    name = name.replace("%", "percent")  # Replace % with 'percent'
    name = name.replace("/", "per")
    name = re.sub(r"[^\w\s]", "", name)  # Remove any non-alphanumeric characters
    
    # Convert numbers in the column name to words
    name = re.sub(r'\d+', lambda x: num2words(x.group(), lang='en'), name).replace('-', '_')
    
    if len(name)>63:
        name = shorten_column_name(name)
    
    name = name.strip().lower().replace(" ", "_")
    # Check if the name contains SQL keywords and replace them
    if name in SQL_KEYWORDS:
        name = name.replace(name, f"sql_{name}")
    
    return name

def infer_column_types(csv_file_path, encoding="utf-8"):
    """
    Infer PostgreSQL-compatible column types from the CSV file.
    - Assumes all values in a column are of the same type.
    - Omits the first column from analysis.
    """
    with open(csv_file_path, 'r', encoding=encoding) as file:
        reader = csv.reader(file)
        header = next(reader)  # Read the header row
        sample_row = next(reader)  # Read the first data row to infer types

        # Omit the first column (index 0)
        header = header[1:]
        sample_row = sample_row[1:]

        column_types = []
        for value in sample_row:
            if value.isdigit():
                column_types.append("INT")
            elif is_float(value):
                column_types.append("FLOAT")
            elif is_date(value):
                column_types.append("DATE")
            else:
                column_types.append("VARCHAR")
        
        # Format column names
        formatted_header = [format_column_name(col) for col in header]
        
        return formatted_header, column_types

def is_float(value):
    """Check if a string represents a float."""
    try:
        float(value)
        return True
    except ValueError:
        return False

def is_date(value):
    """Check if a string represents a date."""
    import dateutil.parser
    try:
        dateutil.parser.parse(value)
        return True
    except ValueError:
        return False

def create_table_from_csv(cursor, table_name, csv_file_path):
    """Create a table dynamically based on the CSV header and inferred types, excluding the first column."""
    header, column_types = infer_column_types(csv_file_path)

    # Construct the CREATE TABLE query
    columns = ", ".join(f"{col} {dtype}" for col, dtype in zip(header, column_types))
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        {columns}
    );
    """
    logger.info("Creating table with query: %s", create_table_query)
    cursor.execute(create_table_query)
    logger.info(f"Table '{table_name}' created successfully with columns: {header}")

def insert_data_from_csv(cursor, table_name, csv_file_path, encoding="utf-8"):
    """Insert data from a CSV file into the dynamically created table, omitting the first column."""
    with open(csv_file_path, 'r', encoding=encoding) as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip the header row

        # Omit the first column (index 0)
        header = header[1:]

        # Format column names
        formatted_header = [format_column_name(col) for col in header]

        # Construct INSERT INTO query
        placeholders = ", ".join(["%s"] * len(formatted_header))
        insert_query = f"INSERT INTO {table_name} ({', '.join(formatted_header)}) VALUES ({placeholders})"

        for row in reader:
            # Omit the first column (index 0)
            row = row[1:]
            cursor.execute(insert_query, row)

    logger.info(f"Data inserted into table '{table_name}' successfully.")

def close_connection(conn, cursor):
    """Close the database connection and cursor."""
    if cursor:
        cursor.close()
    if conn:
        conn.close()
    logger.info("Database connection closed.")

def main():
    """Main function to execute the script."""
    conn = None
    cursor = None
    try:
        # Connect to the database
        conn = connect_to_db(DBNAME, DBUSER, DBPASS, DBHOST, DBPORT)
        cursor = conn.cursor()
        logger.info("Connected to server")

        # Get CSV file path from the user
        csv_file_path = input("Enter the full path to the CSV file: ")
        table_name = input("Enter the table name: ")

        # Create the table dynamically based on the CSV file
        create_table_from_csv(cursor, table_name, csv_file_path)
        conn.commit()

        # Insert data from the CSV
        insert_data_from_csv(cursor, table_name, csv_file_path)
        conn.commit()

    except Exception as e:
        logger.error("An unexpected error occurred: %s", e)
        close_connection(conn, cursor)

    finally:
        # Close the connection
        close_connection(conn, cursor)

if __name__ == "__main__":
    main()
