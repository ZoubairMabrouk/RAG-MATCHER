
import os
import sys
from typing import Optional

import psycopg2
from psycopg2.extras import execute_values


def setup_test_database(connection_string: str) -> None:
    """Set up test database with sample data.

    This function is idempotent and safe to re-run. It will:
      - Ensure required extensions exist (pgcrypto for gen_random_uuid)
      - Create tables: customers, products, orders
      - Create indexes and foreign keys
      - Insert a small set of sample rows if the tables are empty
    """
    print("🔧 Setting up test database...")

    conn = None
    try:
        conn = psycopg2.connect(connection_string)
        conn.autocommit = False

        with conn.cursor() as cur:
            # 1) Ensure required extensions
            cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")

            # 2) Create tables (if not exists)
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS customers (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    email VARCHAR(255) NOT NULL UNIQUE,
                    first_name VARCHAR(100) NOT NULL,
                    last_name  VARCHAR(100) NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS products (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name VARCHAR(255) NOT NULL,
                    price NUMERIC(10,2) NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS orders (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    customer_id UUID NOT NULL,
                    total NUMERIC(10,2) NOT NULL,
                    status VARCHAR(50) NOT NULL DEFAULT 'pending',
                    order_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT fk_orders_customer
                        FOREIGN KEY (customer_id) REFERENCES customers(id)
                        ON DELETE CASCADE
                );
                """
            )

            # 3) Helpful indexes
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_customers_email
                    ON customers (email);
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_products_name
                    ON products (name);
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_orders_customer_id
                    ON orders (customer_id);
                """
            )

            # 4) Seed data if empty
            # Customers
            cur.execute("SELECT COUNT(*) FROM customers;")
            (customer_count,) = cur.fetchone()
            if customer_count == 0:
                customers = [
                    ("alice@example.com", "Alice", "Duran"),
                    ("bob@example.com", "Bob", "Martin"),
                    ("carol@example.com", "Carol", "Ng"),
                ]
                execute_values(
                    cur,
                    """
                    INSERT INTO customers (email, first_name, last_name)
                    VALUES %s
                    ON CONFLICT (email) DO NOTHING
                    """,
                    customers,
                )
                print(f"👤 Inserted {len(customers)} customers.")

            # Products
            cur.execute("SELECT COUNT(*) FROM products;")
            (product_count,) = cur.fetchone()
            if product_count == 0:
                products = [
                    ("Wireless Sensor", 79.99),
                    ("Smart Hub", 149.00),
                    ("Door Controller", 59.50),
                    ("Energy Monitor", 129.90),
                ]
                execute_values(
                    cur,
                    """
                    INSERT INTO products (name, price)
                    VALUES %s
                    ON CONFLICT DO NOTHING
                    """,
                    products,
                )
                print(f"📦 Inserted {len(products)} products.")

            # Orders (only if there are customers and no orders)
            cur.execute("SELECT COUNT(*) FROM orders;")
            (order_count,) = cur.fetchone()
            if order_count == 0:
                # Pick some existing customer ids
                cur.execute("SELECT id FROM customers ORDER BY created_at ASC;")
                customer_ids = [row[0] for row in cur.fetchall()]
                if customer_ids:
                    orders = [
                        (customer_ids[0], 229.99, "paid"),
                        (customer_ids[min(1, len(customer_ids) - 1)], 79.99, "pending"),
                        (customer_ids[min(2, len(customer_ids) - 1)], 188.50, "shipped"),
                    ]
                    execute_values(
                        cur,
                        """
                        INSERT INTO orders (customer_id, total, status)
                        VALUES %s
                        """,
                        orders,
                    )
                    print(f"🧾 Inserted {len(orders)} orders.")

        conn.commit()
        print("✅ Test database is ready.")
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"❌ Failed to set up test database: {e}")
        raise
    finally:
        if conn:
            conn.close()


def _get_connection_string_from_args(argv: list[str]) -> Optional[str]:
    """Resolve a connection string from CLI args or env.

    Precedence:
      1. First positional argument
      2. DATABASE_URL environment variable
    """
    if len(argv) > 1 and argv[1]:
        return argv[1]
    env_url = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
    return env_url


if __name__ == "__main__":
    # dsn = _get_connection_string_from_args(sys.argv)
    # if not dsn:
    #     print(
    #         "Usage: python -m scripts.setup_test_db <postgresql://user:pass@host:port/dbname>\n"
    #         "Or set DATABASE_URL environment variable."
    #     )
    #     sys.exit(2)

    setup_test_database("postgresql://odoo:odoo@localhost:51664/mimic")
