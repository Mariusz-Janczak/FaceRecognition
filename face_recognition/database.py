import sqlite3
import numpy as np
from numpy.linalg import norm
from typing import Optional, Tuple, List


class FaceDatabase:
    def __init__(self, db_path: str = "faces.db"):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._create_tables()

    def _create_tables(self):
        """Create new tabled if do not exist."""
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT
        );
        """)
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            source_image TEXT,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (person_id) REFERENCES persons(id)
        );
        """)
        self.conn.commit()

    # ----------------------
    # Persons
    # ----------------------
    def add_person(self, name: str, description: Optional[str] = None) -> int:
        """Adds new person into database and returns its ID."""
        self.cursor.execute(
            "INSERT INTO persons (name, description) VALUES (?, ?)",
            (name, description),
        )
        self.conn.commit()
        return self.cursor.lastrowid

    def get_person(self, person_id: int) -> Optional[Tuple[int, str, str]]:
        """Get person information."""
        self.cursor.execute("SELECT * FROM persons WHERE id = ?", (person_id,))
        return self.cursor.fetchone()

    # ----------------------
    # Faces
    # ----------------------
    def add_face(self, person_id: int, embedding: np.ndarray, source_image: Optional[str] = None) -> int:
        """Adds face embedding into database."""
        emb_bytes = embedding.astype(np.float32).tobytes()
        self.cursor.execute(
            "INSERT INTO faces (person_id, embedding, source_image) VALUES (?, ?, ?)",
            (person_id, emb_bytes, source_image),
        )
        self.conn.commit()
        return self.cursor.lastrowid

    def load_embeddings(self, person_id: Optional[int] = None) -> List[Tuple[int, int, np.ndarray]]:
        """Gets embeddings from database. If person ID is given there are returned embeddings for the person only."""
        if person_id:
            self.cursor.execute("SELECT id, person_id, embedding FROM faces WHERE person_id = ?", (person_id,))
        else:
            self.cursor.execute("SELECT id, person_id, embedding FROM faces")
        rows = self.cursor.fetchall()
        embeddings = []
        for face_id, pid, emb_bytes in rows:
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            embeddings.append((face_id, pid, emb))
        return embeddings

    # ----------------------
    # Comparizons
    # ----------------------
    @staticmethod
    def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate distance between embeddings."""
        return 1 - np.dot(a, b) / (norm(a) * norm(b))

    def identify_face(self, new_emb: np.ndarray, threshold: float = 0.4) -> Tuple[Optional[str], float]:
        """Face recognition."""
        embeddings = self.load_embeddings()
        best_match = None
        best_dist = float("inf")

        for _, person_id, emb in embeddings:
            dist = self.cosine_distance(new_emb, emb)
            if dist < best_dist:
                best_dist = dist
                best_match = person_id

        if best_match is not None and best_dist < threshold:
            self.cursor.execute("SELECT name FROM persons WHERE id = ?", (best_match,))
            row = self.cursor.fetchone()
            name = row[0] if row else None
            return name, best_dist

        return None, best_dist

    def close(self):
        """Close database connection."""
        self.conn.close()
