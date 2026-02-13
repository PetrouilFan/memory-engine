"""
Database connection pool and optimizations for SQLite.
"""
import os
import sqlite3
import logging
import threading
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)

# Configuration
DB_PATH = os.getenv('MEMORY_DB_PATH', 'data/memory.db')
DB_POOL_SIZE = int(os.getenv('DB_POOL_SIZE', '5'))
DB_TIMEOUT = float(os.getenv('DB_TIMEOUT', '30.0'))
DB_CHECK_SAME_THREAD = os.getenv('DB_CHECK_SAME_THREAD', 'False').lower() == 'true'
ENABLE_WAL = os.getenv('DB_ENABLE_WAL', 'true').lower() == 'true'


class ConnectionPool:
    """Thread-safe SQLite connection pool."""
    
    def __init__(self, db_path: str, pool_size: int = 5, timeout: float = 30.0):
        self.db_path = db_path
        self.timeout = timeout
        self._pool = []
        self._lock = threading.Lock()
        self._semaphore = threading.Semaphore(pool_size)
        
        # Initialize pool
        for _ in range(pool_size):
            conn = self._create_connection()
            self._pool.append(conn)
        
        logger.info(f"Connection pool initialized: {pool_size} connections")
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection with optimizations."""
        conn = sqlite3.connect(
            self.db_path,
            timeout=self.timeout,
            check_same_thread=DB_CHECK_SAME_THREAD,
        )
        # Enable WAL mode for better concurrent performance
        if ENABLE_WAL:
            conn.execute("PRAGMA journal_mode=WAL")
        # Optimize for performance
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA mmap_size=268435456")  # 256MB memory-mapped I/O
        return conn
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool."""
        self._semaphore.acquire()
        conn = None
        try:
            with self._lock:
                conn = self._pool.pop()
            
            # Check if connection is still valid
            try:
                conn.execute("SELECT 1")
            except sqlite3.Error:
                conn = self._create_connection()
            
            yield conn
        finally:
            if conn:
                with self._lock:
                    self._pool.append(conn)
            self._semaphore.release()
    
    def close_all(self):
        """Close all connections in the pool."""
        with self._lock:
            for conn in self._pool:
                try:
                    conn.close()
                except Exception as e:
                    logger.warning(f"Error closing connection: {e}")
            self._pool.clear()
        logger.info("Connection pool closed")


# Global connection pool
_pool: Optional[ConnectionPool] = None
_pool_lock = threading.Lock()


def get_connection_pool() -> ConnectionPool:
    """Get the global connection pool."""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ConnectionPool(DB_PATH, DB_POOL_SIZE, DB_TIMEOUT)
    return _pool


@contextmanager
def get_db_connection():
    """Context manager for database connections using the pool."""
    pool = get_connection_pool()
    with pool.get_connection() as conn:
        yield conn


def init_db_connection(db_path: str = None) -> sqlite3.Connection:
    """Initialize a single database connection (legacy compatibility)."""
    if db_path:
        global DB_PATH
        DB_PATH = db_path
    
    conn = sqlite3.connect(
        DB_PATH,
        timeout=DB_TIMEOUT,
        check_same_thread=False,
    )
    
    if ENABLE_WAL:
        conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA mmap_size=268435456")
    
    return conn
