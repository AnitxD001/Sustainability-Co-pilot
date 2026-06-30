"""
Database Connection
====================
Reads DATABASE_URL from the environment. Targets PostgreSQL in production:

    DATABASE_URL=postgresql://user:password@host:5432/sustainability_db

Falls back to a local SQLite file (./data/sustainability.db) when no
DATABASE_URL is set, so the project runs out of the box without requiring a
running Postgres server.
"""

import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SQLITE_PATH = PROJECT_ROOT / "data" / "sustainability.db"
DEFAULT_SQLITE_PATH.parent.mkdir(exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_SQLITE_PATH}")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)


def init_db():
    """Create all tables. Import schema first so models register with Base."""
    from database import schema  # noqa: F401
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency: yields a session and ensures it's closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_session():
    """Plain context-free session getter for scripts (generator, ml modules)."""
    return SessionLocal()
