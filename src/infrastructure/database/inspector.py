"""Database introspection services."""
from typing import List, Dict, Any 
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
from abc import ABC, abstractmethod
from src.domain.entities.schema import SchemaMetadata, Table, Column, ForeignKey, Index

class IDataBaseInspector(ABC):
    """Interface for database inspection."""
    
    @abstractmethod
    def introspect_schema(self) -> SchemaMetadata:
        """Introspect complete database schema."""
        pass
    
    @abstractmethod
    def get_table_info(self, table_name: str) -> Table:
        """Get detailed information about a table."""
        pass


class PostgresInspector(IDataBaseInspector):
    """
    PostgreSQL database inspector.
    Single Responsibility: Database introspection.
    """
    
    def __init__(self, connection_string: str):
        self._conn_string = connection_string
    
    def introspect_schema(self) -> SchemaMetadata:
        """Introspect complete PostgreSQL schema."""
        conn = psycopg2.connect(self._conn_string)
        
        try:
            tables = self._get_all_tables(conn)
            views = self._get_all_views(conn)
            
            return SchemaMetadata(
                tables=tables,
                views=views,
                database_name=self._get_database_name(conn),
                version=self._get_version(conn),
                introspection_timestamp=datetime.now()
            )
        finally:
            conn.close()
    
    def get_table_info(self, table_name: str) -> Table:
        """Get detailed information about a specific table."""
        conn = psycopg2.connect(self._conn_string)
        
        try:
            columns = self._get_columns(conn, table_name)
            fks = self._get_foreign_keys(conn, table_name)
            indexes = self._get_indexes(conn, table_name)
            stats = self._get_table_stats(conn, table_name)
            
            return Table(
                name=table_name,
                columns=columns,
                foreign_keys=fks,
                indexes=indexes,
                row_count=stats.get("row_count", 0),
                comment=stats.get("comment")
            )
        finally:
            conn.close()
    
    def _get_all_tables(self, conn) -> List[Table]:
        """Get all tables in the database."""
        query = """
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
            AND table_type = 'BASE TABLE'
            ORDER BY table_schema, table_name
        """
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            rows = cur.fetchall()
        
        tables = []
        for row in rows:
            table = self.get_table_info(row['table_name'])
            table.schema = row['table_schema']
            tables.append(table)
        
        return tables
    
    def _get_columns(self, conn, table_name: str) -> List[Column]:
        """Get columns for a table."""
        query = """
            SELECT 
                column_name,
                data_type,
                is_nullable,
                column_default,
                character_maximum_length
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position
        """
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (table_name,))
            rows = cur.fetchall()
        
        columns = []
        for row in rows:
            data_type = row['data_type']
            if row['character_maximum_length']:
                data_type += f"({row['character_maximum_length']})"
            
            columns.append(Column(
                name=row['column_name'],
                data_type=data_type,
                nullable=row['is_nullable'] == 'YES',
                default_value=row['column_default']
            ))
        
        return columns
    
    def _get_foreign_keys(self, conn, table_name: str) -> List[ForeignKey]:
        """Get foreign keys for a table."""
        query = """
            SELECT
                tc.constraint_name,
                kcu.column_name,
                ccu.table_name AS referenced_table,
                ccu.column_name AS referenced_column,
                rc.update_rule,
                rc.delete_rule
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage AS ccu
                ON ccu.constraint_name = tc.constraint_name
            JOIN information_schema.referential_constraints AS rc
                ON rc.constraint_name = tc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY'
                AND tc.table_name = %s
        """
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (table_name,))
            rows = cur.fetchall()
        
        return [
            ForeignKey(
                name=row['constraint_name'],
                column=row['column_name'],
                referenced_table=row['referenced_table'],
                referenced_column=row['referenced_column'],
                on_update=row['update_rule'],
                on_delete=row['delete_rule']
            )
            for row in rows
        ]
    
    def _get_indexes(self, conn, table_name: str) -> List[Index]:
        """Get indexes for a table."""
        query = """
            SELECT
                i.relname AS index_name,
                a.attname AS column_name,
                ix.indisunique AS is_unique,
                am.amname AS index_type
            FROM pg_class t
            JOIN pg_index ix ON t.oid = ix.indrelid
            JOIN pg_class i ON i.oid = ix.indexrelid
            JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(ix.indkey)
            JOIN pg_am am ON i.relam = am.oid
            WHERE t.relname = %s
            ORDER BY i.relname, a.attnum
        """
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (table_name,))
            rows = cur.fetchall()
        
        # Group by index name
        indexes_dict = {}
        for row in rows:
            idx_name = row['index_name']
            if idx_name not in indexes_dict:
                indexes_dict[idx_name] = {
                    'columns': [],
                    'unique': row['is_unique'],
                    'type': row['index_type']
                }
            indexes_dict[idx_name]['columns'].append(row['column_name'])
        
        return [
            Index(
                name=name,
                columns=data['columns'],
                unique=data['unique'],
                index_type=data['type']
            )
            for name, data in indexes_dict.items()
        ]
    
    def _get_table_stats(self, conn, table_name: str) -> Dict[str, Any]:
        """Get table statistics."""
        query = """
            SELECT 
                n_live_tup as row_count,
                obj_description(to_regclass(%s)::oid) as comment
            FROM pg_stat_user_tables
            WHERE relname = %s
        """
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (table_name, table_name))
            row = cur.fetchone()
        
        return dict(row) if row else {}
    
    def _get_all_views(self, conn) -> List[Dict[str, Any]]:
        """Get all views."""
        query = """
            SELECT table_name, view_definition
            FROM information_schema.views
            WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
        """
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            return [dict(row) for row in cur.fetchall()]
    
    def _get_database_name(self, conn) -> str:
        """Get database name."""
        with conn.cursor() as cur:
            cur.execute("SELECT current_database()")
            return cur.fetchone()[0]
    
    def _get_version(self, conn) -> str:
        """Get PostgreSQL version."""
        with conn.cursor() as cur:
            cur.execute("SELECT version()")
            return cur.fetchone()[0]
