from __future__ import annotations

import os
import secrets
import datetime as dt
from typing import Optional, List
import json
from pathlib import Path
import mimetypes

import jwt
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status, Form
from fastapi.responses import Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from sqlmodel import Field, Session, SQLModel, create_engine, select

from .captioner import DEFAULT_MODEL, process_image_bytes, resolve_ml_device

# --- Config ---
API_PREFIX = "/api"
MODEL_NAME = os.getenv("CAPTION_MODEL", DEFAULT_MODEL)
DEVICE_PREFERENCE = os.getenv("CAPTION_DEVICE", "auto")
JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_hex(32))
JWT_ALGO = "HS256"
ADMIN_EMAIL = "abuhanifah@live.com"
ADMIN_PASSWORD = "abu@hanifah.com"
DB_URL = os.getenv("DB_URL", "sqlite:///./image_narrator.db")
MEDIA_ROOT = Path(os.getenv("MEDIA_ROOT", "media"))

# --- App setup ---
app = FastAPI(title="Image Narrator Backend", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
engine = create_engine(DB_URL, connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {})


# --- Models ---
class Role:
    OBSERVER = "observer"
    SUPERVISOR = "supervisor"
    ADMIN = "admin"


class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(index=True, unique=True)
    hashed_password: Optional[str] = None
    role: str = Field(index=True)
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.utcnow())


class Entry(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    narrative: str
    caption: Optional[str] = None
    stats_json: Optional[str] = None
    image_bytes: Optional[bytes] = Field(default=None)
    image_path: Optional[str] = Field(default=None, index=True)
    image_mime: Optional[str] = None
    image_size: Optional[int] = None
    site: Optional[str] = None
    location: Optional[str] = None  # GPS or area text
    tags: Optional[str] = None  # comma-separated
    severity: Optional[str] = None  # e.g., minor/major/critical
    status: str = Field(default="open", index=True)
    assignee: Optional[str] = None
    notes: Optional[str] = None
    closure_notes: Optional[str] = None
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.utcnow(), index=True)
    updated_at: dt.datetime = Field(default_factory=lambda: dt.datetime.utcnow(), index=True)


def ensure_entry_columns():
    """
    Lightweight migration for SQLite: add new columns if the existing table is missing them.
    """
    if not DB_URL.startswith("sqlite"):
        return
    with engine.connect() as conn:
        cols = conn.exec_driver_sql("PRAGMA table_info(entry);").fetchall()
        col_names = {c[1] for c in cols}
        alters = []
        def add(col, type_sql, default=""):
            if col not in col_names:
                alters.append(f"ALTER TABLE entry ADD COLUMN {col} {type_sql} {default};")
        add("site", "TEXT")
        add("location", "TEXT")
        add("tags", "TEXT")
        add("severity", "TEXT")
        add("status", "TEXT", "DEFAULT 'open'")
        add("assignee", "TEXT")
        add("notes", "TEXT")
        add("closure_notes", "TEXT")
        add("updated_at", "DATETIME", f"DEFAULT '{dt.datetime.utcnow()}'")
        add("image_path", "TEXT")
        add("image_size", "INTEGER")
        for stmt in alters:
            conn.exec_driver_sql(stmt)
        conn.commit()


# --- DB init ---
def create_db_and_seed():
    SQLModel.metadata.create_all(engine)
    ensure_entry_columns()
    MEDIA_ROOT.mkdir(parents=True, exist_ok=True)
    with Session(engine) as session:
        admin = session.exec(select(User).where(User.email == ADMIN_EMAIL)).first()
        if not admin:
            hashed = pwd_context.hash(ADMIN_PASSWORD)
            session.add(User(email=ADMIN_EMAIL, hashed_password=hashed, role=Role.ADMIN))
            session.commit()
        else:
            # Re-hash admin password if missing or using an old scheme.
            needs_update = not admin.hashed_password or not pwd_context.verify(
                ADMIN_PASSWORD, admin.hashed_password
            )
            if needs_update:
                admin.hashed_password = pwd_context.hash(ADMIN_PASSWORD)
                admin.role = Role.ADMIN
                session.add(admin)
                session.commit()
        # Ensure an observer row exists if previously created without password; no-op otherwise.


create_db_and_seed()


# --- Auth helpers ---
def create_token(user: User) -> str:
    payload = {
        "sub": str(user.id),
        "email": user.email,
        "role": user.role,
        "exp": dt.datetime.utcnow() + dt.timedelta(hours=12),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(security),
) -> User:
    if creds is None or not creds.scheme.lower() == "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    token = creds.credentials
    try:
        data = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
    except jwt.PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    user_id = int(data.get("sub"))
    with Session(engine) as session:
        user = session.get(User, user_id)
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
        return user


def require_roles(required_roles: set[str]):
    def _dep(user: User = Depends(get_current_user)) -> User:
        if user.role not in required_roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
        return user

    return _dep


def get_or_create_observer(email: str) -> User:
    normalized = email.strip().lower()
    if "@" not in normalized or not normalized:
        raise HTTPException(status_code=400, detail="Valid email is required.")
    with Session(engine) as session:
        existing = session.exec(select(User).where(User.email == normalized)).first()
        if existing:
            return existing
        user = User(email=normalized, hashed_password=None, role=Role.OBSERVER)
        session.add(user)
        session.commit()
        session.refresh(user)
        return user


def _validate_email(email: str) -> str:
    normalized = email.strip().lower()
    if "@" not in normalized or not normalized:
        raise HTTPException(status_code=400, detail="Invalid email.")
    return normalized


# --- Schemas ---
class RegisterRequest(SQLModel):
    email: str
    password: str


class LoginRequest(SQLModel):
    email: str
    password: str


class AdminCreateRequest(SQLModel):
    email: str
    password: str


class AdminResponse(SQLModel):
    id: int
    email: str
    created_at: dt.datetime


class EntryResponse(SQLModel):
    id: int
    user_id: int
    user_email: str
    narrative: str
    caption: Optional[str]
    image_path: Optional[str]
    image_mime: Optional[str]
    image_size: Optional[int]
    created_at: dt.datetime
    site: Optional[str]
    location: Optional[str]
    tags: Optional[str]
    severity: Optional[str]
    status: str
    assignee: Optional[str]
    notes: Optional[str]
    closure_notes: Optional[str]
    updated_at: dt.datetime


class SummaryResponse(SQLModel):
    period: str
    count: int


class UpdateEntryRequest(SQLModel):
    status: Optional[str] = None
    assignee: Optional[str] = None
    severity: Optional[str] = None
    tags: Optional[str] = None
    notes: Optional[str] = None
    closure_notes: Optional[str] = None


# --- Routes ---
@app.get(f"{API_PREFIX}/health")
def health() -> dict:
    device: Optional[str]
    try:
        dev_obj = resolve_ml_device(DEVICE_PREFERENCE)
        device = str(dev_obj) if dev_obj else None
    except Exception as exc:  # pragma: no cover - env dependent
        device = f"unresolved ({exc})"
    return {"status": "ok", "model": MODEL_NAME, "device": device}


@app.post(f"{API_PREFIX}/register")
def register(body: RegisterRequest):
    email = body.email.strip().lower()
    if "@" not in email:
        raise HTTPException(status_code=400, detail="Invalid email.")
    with Session(engine) as session:
        existing = session.exec(select(User).where(User.email == email)).first()
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered.")
        hashed = pwd_context.hash(body.password)
        user = User(email=email, hashed_password=hashed, role=Role.OBSERVER)
        session.add(user)
        session.commit()
        session.refresh(user)
        token = create_token(user)
        return {"token": token, "role": user.role, "email": user.email}


@app.post(f"{API_PREFIX}/login")
def login(body: LoginRequest):
    email = body.email.strip().lower()
    with Session(engine) as session:
        user = session.exec(select(User).where(User.email == email)).first()
        if not user or not user.hashed_password or not pwd_context.verify(body.password, user.hashed_password):
            raise HTTPException(status_code=401, detail="Invalid credentials.")
        if user.role not in {Role.ADMIN, Role.SUPERVISOR}:
            raise HTTPException(status_code=403, detail="Admin or supervisor only.")
        token = create_token(user)
        return {"token": token, "role": user.role, "email": user.email}


@app.get(f"{API_PREFIX}/admins", response_model=List[AdminResponse])
def list_admins(current_user: User = Depends(require_roles({Role.ADMIN}))):
    with Session(engine) as session:
        admins = session.exec(select(User).where(User.role == Role.ADMIN).order_by(User.created_at.desc())).all()
        return [AdminResponse(id=a.id, email=a.email, created_at=a.created_at) for a in admins]


@app.post(f"{API_PREFIX}/admins", response_model=AdminResponse)
def create_admin(body: AdminCreateRequest, current_user: User = Depends(require_roles({Role.ADMIN}))):
    email = _validate_email(body.email)
    if not body.password or len(body.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")
    with Session(engine) as session:
        existing = session.exec(select(User).where(User.email == email)).first()
        if existing:
            raise HTTPException(status_code=400, detail="Email already exists.")
        hashed = pwd_context.hash(body.password)
        user = User(email=email, hashed_password=hashed, role=Role.ADMIN)
        session.add(user)
        session.commit()
        session.refresh(user)
        return AdminResponse(id=user.id, email=user.email, created_at=user.created_at)


@app.delete(f"{API_PREFIX}/admins/{{admin_id}}")
def delete_admin(admin_id: int, current_user: User = Depends(require_roles({Role.ADMIN}))):
    with Session(engine) as session:
        admin = session.get(User, admin_id)
        if not admin or admin.role != Role.ADMIN:
            raise HTTPException(status_code=404, detail="Admin not found.")
        if admin.email == current_user.email:
            raise HTTPException(status_code=400, detail="You cannot remove yourself.")
        admin_count = len(session.exec(select(User).where(User.role == Role.ADMIN)).all())
        if admin_count <= 1:
            raise HTTPException(status_code=400, detail="At least one admin must remain.")
        session.delete(admin)
        session.commit()
    return {"status": "deleted", "id": admin_id}


@app.post(f"{API_PREFIX}/caption")
async def caption_image(
    email: str = Form(...),
    site: Optional[str] = Form(None),
    location: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    severity: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")

    observer = get_or_create_observer(email)

    MEDIA_ROOT.mkdir(parents=True, exist_ok=True)
    orig_ext = ""
    if file.filename:
        orig_ext = os.path.splitext(file.filename)[1]
    guessed_ext = ""
    if file.content_type:
        guessed_ext = mimetypes.guess_extension(file.content_type.split(";")[0]) or ""
    ext = orig_ext or guessed_ext or ".bin"
    if not ext.startswith("."):
        ext = f".{ext}"
    unique_name = f"entry_{int(dt.datetime.utcnow().timestamp())}_{secrets.token_hex(6)}{ext}"
    file_path = (MEDIA_ROOT / unique_name).resolve()
    try:
        with open(file_path, "wb") as f:
            f.write(data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to store image: {exc}") from exc

    try:
        narrative, stats, caption = process_image_bytes(
            data, model=MODEL_NAME, device_pref=DEVICE_PREFERENCE
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}") from exc

    entry = Entry(
        user_id=observer.id,
        narrative=narrative,
        caption=caption,
        stats_json=json.dumps(stats),
        image_bytes=b"",  # keep column minimal; actual file stored on disk
        image_path=str(file_path),
        image_mime=file.content_type or "application/octet-stream",
        image_size=len(data),
        site=site,
        location=location,
        tags=tags,
        severity=severity,
        notes=notes,
        status="open",
        updated_at=dt.datetime.utcnow(),
    )
    with Session(engine) as session:
        session.add(entry)
        session.commit()
        session.refresh(entry)

    return {
        "narrative": narrative,
        "caption": caption,
        "stats": stats,
        "device": DEVICE_PREFERENCE,
        "model": MODEL_NAME,
        "entry_id": entry.id,
    }


@app.get(f"{API_PREFIX}/entries", response_model=List[EntryResponse])
def list_entries(current_user: User = Depends(require_roles({Role.ADMIN, Role.SUPERVISOR}))):
    with Session(engine) as session:
        stmt = select(
            Entry.id,
            Entry.user_id,
            Entry.narrative,
            Entry.caption,
            Entry.image_path,
            Entry.image_mime,
            Entry.image_size,
            Entry.created_at,
            Entry.site,
            Entry.location,
            Entry.tags,
            Entry.severity,
            Entry.status,
            Entry.assignee,
            Entry.notes,
            Entry.closure_notes,
            Entry.updated_at,
        ).order_by(Entry.created_at.desc())
        entries = session.exec(stmt).all()
        users = {u.id: u.email for u in session.exec(select(User)).all()}
        responses: List[EntryResponse] = []
        for (
            entry_id,
            user_id,
            narrative,
            caption,
            image_path,
            image_mime,
            image_size,
            created_at,
            site,
            location,
            tags,
            severity,
            status,
            assignee,
            notes,
            closure_notes,
            updated_at,
        ) in entries:
            responses.append(
                EntryResponse(
                    id=entry_id,
                    user_id=user_id,
                    user_email=users.get(user_id, "unknown"),
                    narrative=narrative,
                    caption=caption,
                    image_path=image_path,
                    image_mime=image_mime,
                    image_size=image_size,
                    created_at=created_at,
                    site=site,
                    location=location,
                    tags=tags,
                    severity=severity,
                    status=status,
                    assignee=assignee,
                    notes=notes,
                    closure_notes=closure_notes,
                    updated_at=updated_at,
                )
            )
        return responses


@app.get(f"{API_PREFIX}/summary", response_model=List[SummaryResponse])
def summary(period: str = "daily", current_user: User = Depends(require_roles({Role.ADMIN, Role.SUPERVISOR}))):
    if period not in {"daily", "weekly", "monthly"}:
        raise HTTPException(status_code=400, detail="period must be daily, weekly, or monthly")

    now = dt.datetime.utcnow()
    if period == "daily":
        start = now - dt.timedelta(days=1)
    elif period == "weekly":
        start = now - dt.timedelta(weeks=1)
    else:
        start = now - dt.timedelta(days=30)

    with Session(engine) as session:
        count = len(session.exec(select(Entry).where(Entry.created_at >= start)).all())
    return [SummaryResponse(period=period, count=count)]


@app.patch(f"{API_PREFIX}/entries/{{entry_id}}")
def update_entry(
    entry_id: int,
    body: UpdateEntryRequest,
    current_user: User = Depends(require_roles({Role.ADMIN, Role.SUPERVISOR})),
):
    with Session(engine) as session:
        entry = session.get(Entry, entry_id)
        if not entry:
            raise HTTPException(status_code=404, detail="Entry not found")
        if body.status is not None:
            entry.status = body.status
        if body.assignee is not None:
            entry.assignee = body.assignee
        if body.severity is not None:
            entry.severity = body.severity
        if body.tags is not None:
            entry.tags = body.tags
        if body.notes is not None:
            entry.notes = body.notes
        if body.closure_notes is not None:
            entry.closure_notes = body.closure_notes
        entry.updated_at = dt.datetime.utcnow()
        session.add(entry)
        session.commit()
        session.refresh(entry)
        return {"status": "updated", "id": entry_id}


@app.delete(f"{API_PREFIX}/entries/{{entry_id}}")
def delete_entry(entry_id: int, current_user: User = Depends(require_roles({Role.ADMIN, Role.SUPERVISOR}))):
    with Session(engine) as session:
        entry = session.get(Entry, entry_id)
        if not entry:
            raise HTTPException(status_code=404, detail="Entry not found")
        if entry.image_path:
            file_path = Path(entry.image_path)
            if not file_path.is_absolute():
                file_path = (MEDIA_ROOT / file_path).resolve()
            try:
                file_path.unlink(missing_ok=True)
            except Exception:
                pass
        session.delete(entry)
        session.commit()
    return {"status": "deleted", "id": entry_id}


@app.get(f"{API_PREFIX}/entries/{{entry_id}}/image")
def get_entry_image(entry_id: int, current_user: User = Depends(require_roles({Role.ADMIN, Role.SUPERVISOR}))):
    with Session(engine) as session:
        entry = session.get(Entry, entry_id)
        if not entry:
            raise HTTPException(status_code=404, detail="Entry not found")
        if entry.image_path:
            file_path = Path(entry.image_path)
            if not file_path.is_absolute():
                file_path = (MEDIA_ROOT / file_path).resolve()
            if file_path.exists():
                return FileResponse(file_path, media_type=entry.image_mime or "application/octet-stream")
            raise HTTPException(status_code=404, detail="Image file not found on disk.")
        if entry.image_bytes:
            return Response(content=entry.image_bytes, media_type=entry.image_mime)
        raise HTTPException(status_code=404, detail="Image not available.")


@app.delete(f"{API_PREFIX}/entries")
def delete_all(current_user: User = Depends(require_roles({Role.ADMIN, Role.SUPERVISOR}))):
    with Session(engine) as session:
        session.exec(Entry.__table__.delete())  # type: ignore
        session.commit()
    try:
        for path in MEDIA_ROOT.glob("entry_*"):
            path.unlink(missing_ok=True)
    except Exception:
        pass
    return {"status": "deleted_all"}


# Convenience entrypoint: uvicorn backend.main:app --host 0.0.0.0 --port 8000
