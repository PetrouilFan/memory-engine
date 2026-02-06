"""Optional PostgreSQL backend for memory_engine.

If POSTGRES_URL is set, provides a get_db_connection() using SQLAlchemy
that can replace the default SQLite connection pool.
"""

import os
import json
import logging
from contextlib import contextmanager
from typing import Generator

logger = logging.getLogger(__name__)

try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker
    _SQLALCHEMY_AVAILABLE = True
except ImportError:
    _SQLALCHEMY_AVAILABLE = False
    logger.info("SQLAlchemy not installed; PostgreSQL backend unavailable")

POSTGRES_URL = os.getenv("POSTGRES_URL")

if POSTGRES_URL and _SQLALCHEMY_AVAILABLE:
    engine = create_engine(
        POSTGRES_URL,
        pool_size=int(os.getenv("POSTGRES_POOL_SIZE", "5")),
        max_overflow=10,
        future=True,
    )
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    logger.info("PostgreSQL backend enabled for memory_engine")
else:
    engine = None
    SessionLocal = None
    if not POSTGRES_URL:
        logger.info("PostgreSQL backend not configured; using SQLite fallback")

@contextmanager
def get_db_connection() -> Generator:
    """Yield a SQLAlchemy session if PostgreSQL is configured."""
    if not SessionLocal:
        raise RuntimeError("POSTGRES_URL not set – PostgreSQL backend unavailable")
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"PostgreSQL session error: {e}")
        raise
    finally:
        session.close()
