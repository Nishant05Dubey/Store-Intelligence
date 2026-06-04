"""
Database initialization and connection management.
Uses SQLite with WAL mode for concurrent reads during writes.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from sqlmodel import SQLModel, create_engine, Session, select
import structlog

logger = structlog.get_logger(__name__)

DB_PATH = os.getenv("DB_PATH", "/data/store_intelligence.db")
POS_CSV_PATH = os.getenv("POS_CSV_PATH", "/data/pos_transactions.csv")
STORE_LAYOUT_PATH = os.getenv("STORE_LAYOUT_PATH", "/data/store_layout.json")

# Can be overridden in tests
_engine = None
_override_engine = None


def set_engine(engine):
    """Override the engine (used in tests)."""
    global _override_engine
    _override_engine = engine


def clear_engine():
    """Clear engine override."""
    global _override_engine, _engine
    _override_engine = None
    _engine = None


def get_engine():
    if _override_engine is not None:
        return _override_engine
    global _engine
    if _engine is None:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        db_url = f"sqlite:///{DB_PATH}"
        _engine = create_engine(
            db_url,
            echo=False,
            connect_args={"check_same_thread": False},
        )
        # Enable WAL mode for concurrent reads
        with _engine.connect() as conn:
            conn.exec_driver_sql("PRAGMA journal_mode=WAL")
            conn.exec_driver_sql("PRAGMA synchronous=NORMAL")
            conn.exec_driver_sql("PRAGMA cache_size=-64000")
            conn.exec_driver_sql("PRAGMA foreign_keys=ON")
    return _engine


def create_tables():
    """Create all database tables."""
    from app.models import EventRow, PosTransactionRow  # noqa: F401
    SQLModel.metadata.create_all(get_engine())
    logger.info("db.tables_created")


def get_session():
    """FastAPI dependency for DB sessions."""
    with Session(get_engine()) as session:
        yield session


def load_pos_transactions():
    """
    Load POS transaction data from CSV into the database.
    Idempotent — skips rows that already exist.
    Uses the actual Brigade Bangalore POS data.
    """
    import pandas as pd
    from app.models import PosTransactionRow

    if not Path(POS_CSV_PATH).exists():
        logger.warning("pos.csv_not_found", path=POS_CSV_PATH)
        return

    df = pd.read_csv(POS_CSV_PATH)

    # Map CSV columns to our schema
    # order_id = transaction, order_date + order_time = timestamp, GMV = basket
    # We deduplicate by order_id (each order may have multiple line items)
    orders = (
        df.groupby("order_id")
        .agg(
            order_date=("order_date", "first"),
            order_time=("order_time", "first"),
            store_id=("store_id", "first"),
            customer_number=("customer_number", "first"),
            total_amount=("total_amount", "sum"),
        )
        .reset_index()
    )

    store_layout = _load_store_layout()
    pos_store_id = store_layout.get("pos_store_id", "ST1008")
    our_store_id = store_layout.get("store_id", "STORE_BLR_001")

    inserted = 0
    skipped = 0
    with Session(get_engine()) as session:
        for _, row in orders.iterrows():
            txn_id = f"TXN_{row['order_id']}"

            # Check if already exists
            existing = session.get(PosTransactionRow, txn_id)
            if existing:
                skipped += 1
                continue

            # Parse timestamp
            try:
                date_str = str(row["order_date"]).strip()
                time_str = str(row["order_time"]).strip()
                # Format: 10-04-2026 and 16:55:36
                dt = datetime.strptime(
                    f"{date_str} {time_str}", "%d-%m-%Y %H:%M:%S"
                )
                ts = dt.isoformat() + "Z"
            except Exception:
                ts = datetime.utcnow().isoformat() + "Z"

            # Only load transactions for our store
            if str(row["store_id"]) != pos_store_id:
                continue

            txn = PosTransactionRow(
                transaction_id=txn_id,
                store_id=our_store_id,
                timestamp=ts,
                basket_value=float(row["total_amount"] or 0),
                customer_number=str(row.get("customer_number", "")) or None,
            )
            session.add(txn)
            inserted += 1

        session.commit()

    logger.info("pos.loaded", inserted=inserted, skipped=skipped)


def _load_store_layout() -> dict:
    if Path(STORE_LAYOUT_PATH).exists():
        with open(STORE_LAYOUT_PATH) as f:
            return json.load(f)
    return {}
