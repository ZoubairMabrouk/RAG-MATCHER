
import pytest
import psycopg2
from testcontainers.postgres import PostgresContainer

@pytest.fixture(scope="session")
def postgres_container():
    """Start PostgreSQL container for testing."""
    with PostgresContainer("postgres:15") as postgres:
        yield postgres


@pytest.fixture
def db_connection(postgres_container):
    """Get database connection."""
    conn = psycopg2.connect(postgres_container.get_connection_url())
    yield conn
    conn.close()


@pytest.fixture
def clean_database(db_connection):
    """Ensure clean database for each test."""
    with db_connection.cursor() as cur:
        # Drop all tables
        cur.execute("""
            DROP SCHEMA public CASCADE;
            CREATE SCHEMA public;
        """)
    db_connection.commit()
    yield db_connection


def test_create_test_schema(clean_database):
    """Test creating a sample schema."""
    with clean_database.cursor() as cur:
        cur.execute("""
            CREATE TABLE customers (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                email VARCHAR(255) NOT NULL UNIQUE,
                first_name VARCHAR(100) NOT NULL,
                last_name VARCHAR(100) NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE products (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name VARCHAR(255) NOT NULL,
                price NUMERIC(10,2) NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
        """)
    clean_database.commit()
    
    # Verify tables exist
    with clean_database.cursor() as cur:
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
        """)
        tables = [row[0] for row in cur.fetchall()]
    
    assert 'customers' in tables
    assert 'products' in tables
