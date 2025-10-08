import os
import sys
from typing import Optional

import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv


def load_connection_string(argv: list[str]) -> Optional[str]:
    """Charge la chaîne de connexion depuis les arguments ou le fichier .env.example"""
    load_dotenv(dotenv_path=".env.example")

    # 1. Argument en ligne de commande
    if len(argv) > 1:
        return argv[1]

    # 2. Variable d'environnement
    return os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")


def setup_test_database(connection_string: str) -> None:
    """Initialise la base de données avec des tables et des données de test."""
    print("🔧 Initialisation de la base de test...")

    try:
        with psycopg2.connect(connection_string) as conn:
            with conn.cursor() as cur:
                # Activer l'extension pgcrypto pour UUID
                cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")

                # Création des tables
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS customers (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        email VARCHAR(255) NOT NULL UNIQUE,
                        first_name VARCHAR(100) NOT NULL,
                        last_name  VARCHAR(100) NOT NULL,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS products (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        name VARCHAR(255) NOT NULL,
                        price NUMERIC(10,2) NOT NULL,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cur.execute("""
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
                """)

                # Index utiles
                cur.execute("CREATE INDEX IF NOT EXISTS idx_customers_email ON customers (email);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_products_name ON products (name);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_orders_customer_id ON orders (customer_id);")

                # Insérer des clients si vide
                cur.execute("SELECT COUNT(*) FROM customers;")
                if cur.fetchone()[0] == 0:
                    customers = [
                        ("alice@example.com", "Alice", "Duran"),
                        ("bob@example.com", "Bob", "Martin"),
                        ("carol@example.com", "Carol", "Ng"),
                    ]
                    execute_values(
                        cur,
                        "INSERT INTO customers (email, first_name, last_name) VALUES %s ON CONFLICT (email) DO NOTHING;",
                        customers
                    )
                    print(f"👤 {len(customers)} clients ajoutés.")

                # Produits
                cur.execute("SELECT COUNT(*) FROM products;")
                if cur.fetchone()[0] == 0:
                    products = [
                        ("Wireless Sensor", 79.99),
                        ("Smart Hub", 149.00),
                        ("Door Controller", 59.50),
                        ("Energy Monitor", 129.90),
                    ]
                    execute_values(
                        cur,
                        "INSERT INTO products (name, price) VALUES %s ON CONFLICT DO NOTHING;",
                        products
                    )
                    print(f"📦 {len(products)} produits ajoutés.")

                # Commandes
                cur.execute("SELECT COUNT(*) FROM orders;")
                if cur.fetchone()[0] == 0:
                    cur.execute("SELECT id FROM customers ORDER BY created_at;")
                    customer_ids = [row[0] for row in cur.fetchall()]

                    if customer_ids:
                        orders = [
                            (customer_ids[0], 229.99, "paid"),
                            (customer_ids[min(1, len(customer_ids) - 1)], 79.99, "pending"),
                            (customer_ids[min(2, len(customer_ids) - 1)], 188.50, "shipped"),
                        ]
                        execute_values(
                            cur,
                            "INSERT INTO orders (customer_id, total, status) VALUES %s;",
                            orders
                        )
                        print(f"🧾 {len(orders)} commandes ajoutées.")

            conn.commit()
            print("✅ Base de test prête.")

    except psycopg2.Error as e:
        print("❌ Erreur psycopg2 :", e.pgerror or e)
        sys.exit(1)
    except Exception as e:
        print("❌ Erreur inattendue :", str(e))
        sys.exit(1)


if __name__ == "__main__":
    dsn = load_connection_string(sys.argv)

    if not dsn:
        print("❌ Aucune chaîne de connexion trouvée.")
        print("Utilisation :")
        print("  python setup_test_db.py <postgresql://user:pass@host:port/db>")
        print("  OU définir DATABASE_URL dans .env.example")
        sys.exit(1)

    setup_test_database(dsn)
