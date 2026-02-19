"""Tests for the inCite web tier (auth, database, Zotero mapping, routes).

These tests require a running PostgreSQL instance with the pgvector extension.
Set DATABASE_URL to enable them. They are skipped in CI without Postgres.

Run locally: DATABASE_URL=postgresql://... pytest tests/test_web_tier.py -v
"""

import os
import tempfile
import uuid
from pathlib import Path

import pytest

# Set required env vars before importing cloud modules
os.environ.setdefault("ENCRYPTION_KEY", "7gRTU890D-m7JaPp6-ks4KMHLpXs5-ugRvtjfAXwZPE=")  # test key
os.environ.setdefault("INVITE_CODE", "test-invite-123")

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set — skipping Postgres-dependent tests",
)


# ---------------------------------------------------------------------------
# Database CRUD
# ---------------------------------------------------------------------------


@pytest.fixture
def pg_db():
    """Get a Postgres connection for testing."""
    from cloud.database import get_db, init_db, return_conn

    init_db()
    db = get_db()
    yield db
    return_conn(db)


def test_create_and_get_user(pg_db):
    from cloud.database import create_user, get_user_by_email, get_user_by_id

    email = f"alice-{uuid.uuid4().hex[:8]}@example.com"
    uid = create_user(pg_db, "Alice", email, "hashed_pw")
    assert uid > 0

    user = get_user_by_email(pg_db, email)
    assert user is not None
    assert user["name"] == "Alice"
    assert user["email"] == email

    user2 = get_user_by_id(pg_db, uid)
    assert user2 is not None
    assert user2["name"] == "Alice"


def test_get_nonexistent_user(pg_db):
    from cloud.database import get_user_by_email

    assert get_user_by_email(pg_db, "nobody@example.com") is None


def test_session_lifecycle(pg_db):
    from cloud.database import (
        create_session,
        create_user,
        delete_session,
        validate_session,
    )

    email = f"bob-{uuid.uuid4().hex[:8]}@example.com"
    uid = create_user(pg_db, "Bob", email, "hashed_pw")
    token = create_session(pg_db, uid)
    assert len(token) == 64  # Two uuid4 hex strings

    # Validate returns user
    user = validate_session(pg_db, token)
    assert user is not None
    assert user["name"] == "Bob"

    # Delete session
    delete_session(pg_db, token)
    assert validate_session(pg_db, token) is None


def test_invalid_session(pg_db):
    from cloud.database import validate_session

    assert validate_session(pg_db, "nonexistent_token") is None


def test_library_crud(pg_db):
    from cloud.database import (
        create_library,
        create_user,
        get_library,
        get_user_libraries,
        update_library,
    )

    email = f"carol-{uuid.uuid4().hex[:8]}@example.com"
    uid = create_user(pg_db, "Carol", email, "hashed_pw")
    lib_id = create_library(pg_db, uid, "12345", "encrypted_key")
    assert lib_id > 0

    lib = get_library(pg_db, lib_id)
    assert lib is not None
    assert lib["zotero_user_id"] == "12345"
    assert lib["status"] == "pending"

    update_library(pg_db, lib_id, status="ready", num_papers=42)
    lib = get_library(pg_db, lib_id)
    assert lib["status"] == "ready"
    assert lib["num_papers"] == 42

    libs = get_user_libraries(pg_db, uid)
    assert len(libs) == 1


def test_update_library_rejects_invalid_columns(pg_db):
    from cloud.database import create_library, create_user, update_library

    email = f"dave-{uuid.uuid4().hex[:8]}@example.com"
    uid = create_user(pg_db, "Dave", email, "hashed_pw")
    lib_id = create_library(pg_db, uid, "99999", "enc")

    with pytest.raises(ValueError, match="Invalid library columns"):
        update_library(pg_db, lib_id, evil_column="drop table users")


def test_processing_job_crud(pg_db):
    from cloud.database import (
        create_library,
        create_processing_job,
        create_user,
        get_library_active_job,
        get_processing_job,
        update_processing_job,
    )

    email = f"eve-{uuid.uuid4().hex[:8]}@example.com"
    uid = create_user(pg_db, "Eve", email, "hashed_pw")
    lib_id = create_library(pg_db, uid, "11111", "enc")

    job_id = create_processing_job(pg_db, lib_id)
    assert len(job_id) == 32

    job = get_processing_job(pg_db, job_id)
    assert job is not None
    assert job["status"] == "pending"

    update_processing_job(pg_db, job_id, status="processing", progress_stage="extracting")
    job = get_processing_job(pg_db, job_id)
    assert job["status"] == "processing"
    assert job["progress_stage"] == "extracting"

    active = get_library_active_job(pg_db, lib_id)
    assert active is not None
    assert active["id"] == job_id


# ---------------------------------------------------------------------------
# Password hashing (no DB needed)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(False, reason="")  # Always runs — no DB dependency
def test_password_hashing():
    from cloud.web_auth import hash_password, verify_password

    hashed = hash_password("my_secure_password")
    assert hashed != "my_secure_password"
    assert verify_password("my_secure_password", hashed)
    assert not verify_password("wrong_password", hashed)


@pytest.mark.skipif(False, reason="")
def test_invite_code_validation():
    from cloud.web_auth import validate_invite_code

    assert validate_invite_code("test-invite-123")
    assert not validate_invite_code("wrong-code")


# ---------------------------------------------------------------------------
# Zotero paper mapping (no DB needed)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(False, reason="")
def test_zotero_to_paper():
    pytest.importorskip("pyzotero")
    from cloud.zotero_web import _zotero_to_paper

    item = {
        "key": "ABC123",
        "data": {
            "key": "ABC123",
            "itemType": "journalArticle",
            "title": "Climate Change Impacts on Coral Reefs",
            "abstractNote": "We study the effects of warming oceans.",
            "date": "2023-05-15",
            "DOI": "10.1234/test",
            "publicationTitle": "Nature Climate Change",
            "citationKey": "smith2023coral",
            "creators": [
                {"creatorType": "author", "firstName": "John", "lastName": "Smith"},
                {"creatorType": "author", "firstName": "Jane", "lastName": "Doe"},
            ],
        },
    }

    paper = _zotero_to_paper(item)
    assert paper is not None
    assert paper.id == "ABC123"
    assert paper.title == "Climate Change Impacts on Coral Reefs"
    assert paper.abstract == "We study the effects of warming oceans."
    assert paper.year == 2023
    assert paper.doi == "10.1234/test"
    assert paper.journal == "Nature Climate Change"
    assert paper.bibtex_key == "smith2023coral"
    assert len(paper.authors) == 2
    assert paper.authors[0] == "John Smith"


@pytest.mark.skipif(False, reason="")
def test_zotero_to_paper_no_title():
    pytest.importorskip("pyzotero")
    from cloud.zotero_web import _zotero_to_paper

    item = {"data": {"key": "XYZ", "title": "", "itemType": "journalArticle"}}
    assert _zotero_to_paper(item) is None


@pytest.mark.skipif(False, reason="")
def test_zotero_to_paper_conference():
    pytest.importorskip("pyzotero")
    from cloud.zotero_web import _zotero_to_paper

    item = {
        "key": "CONF1",
        "data": {
            "key": "CONF1",
            "title": "A Conference Paper",
            "proceedingsTitle": "NeurIPS 2023",
            "date": "2023",
            "creators": [{"creatorType": "author", "name": "Research Group"}],
        },
    }

    paper = _zotero_to_paper(item)
    assert paper is not None
    assert paper.journal == "NeurIPS 2023"
    assert paper.authors == ["Research Group"]


# ---------------------------------------------------------------------------
# User agent manager (no DB needed for basic tests)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(False, reason="")
def test_agent_manager_invalidate():
    from cloud.user_agents import UserAgentManager

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = UserAgentManager(max_cached=5, data_dir=Path(tmpdir))
        manager.invalidate("anything")  # Should not raise
        assert manager.cached_count == 0


# ---------------------------------------------------------------------------
# FastAPI routes (integration — requires Postgres)
# ---------------------------------------------------------------------------


class CSRFTestClient:
    """Wrapper around TestClient that auto-handles CSRF tokens."""

    def __init__(self, client):
        self._client = client
        self._csrf_token = "test-csrf-token-" + "a" * 32

    def get(self, *args, **kwargs):
        resp = self._client.get(*args, **kwargs)
        return resp

    def _add_csrf(self, kwargs):
        """Inject CSRF token into cookies, headers, and form data."""
        cookies = kwargs.get("cookies", {}) or {}
        cookies["csrf_token"] = self._csrf_token
        kwargs["cookies"] = cookies

        headers = kwargs.get("headers", {}) or {}
        headers["X-CSRF-Token"] = self._csrf_token
        kwargs["headers"] = headers

        if "data" in kwargs and isinstance(kwargs["data"], dict):
            kwargs["data"]["csrf_token"] = self._csrf_token

        return kwargs

    def post(self, url, **kwargs):
        return self._client.post(url, **self._add_csrf(kwargs))

    def delete(self, url, **kwargs):
        return self._client.delete(url, **self._add_csrf(kwargs))

    def __getattr__(self, name):
        return getattr(self._client, name)


@pytest.fixture
def test_app(pg_db):
    """Create a test FastAPI app with all routes mounted."""
    from concurrent.futures import ThreadPoolExecutor

    from fastapi.templating import Jinja2Templates
    from fastapi.testclient import TestClient

    from cloud.server import app
    from cloud.user_agents import UserAgentManager

    # Override app state for testing
    template_dir = Path(__file__).parent.parent / "cloud" / "templates"
    app.state.db = pg_db
    app.state.templates = Jinja2Templates(directory=str(template_dir))
    app.state.agent_manager = UserAgentManager(
        max_cached=5,
        data_dir=Path(tempfile.mkdtemp()),
    )
    app.state.executor = ThreadPoolExecutor(max_workers=1)

    client = TestClient(app, follow_redirects=False)
    yield CSRFTestClient(client)


def test_root_shows_landing_page(test_app):
    resp = test_app.get("/")
    assert resp.status_code == 200
    assert "InCite" in resp.text


def test_login_page(test_app):
    resp = test_app.get("/web/login")
    assert resp.status_code == 200
    assert "Sign in" in resp.text


def test_signup_page(test_app):
    resp = test_app.get("/web/signup")
    assert resp.status_code == 200
    assert "Create your account" in resp.text


def test_signup_flow(test_app):
    email = f"test-{uuid.uuid4().hex[:8]}@example.com"
    resp = test_app.post(
        "/web/signup",
        data={
            "name": "Test User",
            "email": email,
            "password": "securepassword123",
            "invite_code": "test-invite-123",
        },
    )
    assert resp.status_code == 303
    assert "/web/onboarding" in resp.headers.get("location", "")
    assert "session" in resp.cookies


def test_signup_bad_invite_code(test_app):
    email = f"bad-{uuid.uuid4().hex[:8]}@example.com"
    resp = test_app.post(
        "/web/signup",
        data={
            "name": "Bad User",
            "email": email,
            "password": "securepassword123",
            "invite_code": "wrong-code",
        },
    )
    assert resp.status_code == 400
    assert "Invalid invite code" in resp.text


def test_login_flow(test_app, pg_db):
    from cloud.database import create_user
    from cloud.web_auth import hash_password

    email = f"login-{uuid.uuid4().hex[:8]}@example.com"
    pw_hash = hash_password("mypassword")
    create_user(pg_db, "Login User", email, pw_hash)

    resp = test_app.post(
        "/web/login",
        data={"email": email, "password": "mypassword"},
    )
    assert resp.status_code == 303
    assert "/web/recommend" in resp.headers.get("location", "")
    assert "session" in resp.cookies


def test_login_bad_password(test_app, pg_db):
    from cloud.database import create_user
    from cloud.web_auth import hash_password

    email = f"auth-{uuid.uuid4().hex[:8]}@example.com"
    pw_hash = hash_password("rightpassword")
    create_user(pg_db, "Auth User", email, pw_hash)

    resp = test_app.post(
        "/web/login",
        data={"email": email, "password": "wrongpassword"},
    )
    assert resp.status_code == 400
    assert "Invalid email or password" in resp.text


def test_recommend_requires_auth(test_app):
    resp = test_app.get("/web/recommend")
    assert resp.status_code == 303
    assert "/web/login" in resp.headers.get("location", "")


def test_recommend_redirects_to_onboarding(test_app, pg_db):
    """Authenticated user without a library should be redirected to onboarding."""
    from cloud.database import create_session, create_user
    from cloud.web_auth import hash_password

    email = f"new-{uuid.uuid4().hex[:8]}@example.com"
    uid = create_user(pg_db, "New User", email, hash_password("pass1234"))
    token = create_session(pg_db, uid)

    resp = test_app.get("/web/recommend", cookies={"session": token})
    assert resp.status_code == 303
    assert "/web/onboarding" in resp.headers.get("location", "")


def test_health_endpoint(test_app):
    resp = test_app.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data


# ---------------------------------------------------------------------------
# Manual library creation
# ---------------------------------------------------------------------------


def test_create_library_manual(pg_db):
    from cloud.database import create_library_manual, create_user, get_library

    email = f"manual-{uuid.uuid4().hex[:8]}@example.com"
    uid = create_user(pg_db, "Manual User", email, "hashed_pw")
    lib_id = create_library_manual(pg_db, uid)
    assert lib_id > 0

    lib = get_library(pg_db, lib_id)
    assert lib is not None
    assert lib["source_type"] == "manual"
    assert lib["zotero_user_id"] is None
    assert lib["zotero_api_key_encrypted"] is None
    assert lib["status"] == "pending"


# ---------------------------------------------------------------------------
# Account, Library, Upload page routes
# ---------------------------------------------------------------------------


def test_account_page_requires_auth(test_app):
    resp = test_app.get("/web/account")
    assert resp.status_code == 303
    assert "/web/login" in resp.headers.get("location", "")


def test_account_page_renders(test_app, pg_db):
    """Authenticated user with a library sees the account page."""
    from cloud.database import create_library_manual, create_session, create_user, update_library
    from cloud.web_auth import hash_password

    email = f"account-{uuid.uuid4().hex[:8]}@example.com"
    uid = create_user(pg_db, "Account User", email, hash_password("pass1234"))
    lib_id = create_library_manual(pg_db, uid)
    update_library(pg_db, lib_id, status="ready", num_papers=10, num_chunks=50)
    token = create_session(pg_db, uid)

    resp = test_app.get("/web/account", cookies={"session": token})
    assert resp.status_code == 200
    assert "Account User" in resp.text
    assert "10" in resp.text  # num_papers


def test_library_page_requires_auth(test_app):
    resp = test_app.get("/web/library")
    assert resp.status_code == 303
    assert "/web/login" in resp.headers.get("location", "")


def test_library_redirects_without_library(test_app, pg_db):
    """Authenticated user without library is redirected to onboarding."""
    from cloud.database import create_session, create_user
    from cloud.web_auth import hash_password

    email = f"nolib-{uuid.uuid4().hex[:8]}@example.com"
    uid = create_user(pg_db, "No Lib User", email, hash_password("pass1234"))
    token = create_session(pg_db, uid)

    resp = test_app.get("/web/library", cookies={"session": token})
    assert resp.status_code == 303
    assert "/web/onboarding" in resp.headers.get("location", "")


def test_upload_page_requires_auth(test_app):
    resp = test_app.get("/web/upload")
    assert resp.status_code == 303
    assert "/web/login" in resp.headers.get("location", "")


def test_upload_page_renders(test_app, pg_db):
    """Authenticated user with a library sees the upload page."""
    from cloud.database import create_library_manual, create_session, create_user, update_library
    from cloud.web_auth import hash_password

    email = f"upload-{uuid.uuid4().hex[:8]}@example.com"
    uid = create_user(pg_db, "Upload User", email, hash_password("pass1234"))
    lib_id = create_library_manual(pg_db, uid)
    update_library(pg_db, lib_id, status="ready")
    token = create_session(pg_db, uid)

    resp = test_app.get("/web/upload", cookies={"session": token})
    assert resp.status_code == 200
    assert "Upload PDFs" in resp.text
    assert "Enter Metadata" in resp.text


def test_onboarding_manual_skip(test_app, pg_db):
    """POST with action=manual creates a manual library and redirects to account."""
    from cloud.database import create_session, create_user, get_user_libraries
    from cloud.web_auth import hash_password

    email = f"skip-{uuid.uuid4().hex[:8]}@example.com"
    uid = create_user(pg_db, "Skip User", email, hash_password("pass1234"))
    token = create_session(pg_db, uid)

    resp = test_app.post(
        "/web/onboarding",
        data={"action": "manual"},
        cookies={"session": token},
    )
    assert resp.status_code == 303, f"Expected 303, got {resp.status_code}: {resp.text[:300]}"
    assert "/web/recommend" in resp.headers.get("location", "")

    # Verify library was created
    libraries = get_user_libraries(pg_db, uid)
    assert len(libraries) == 1
    assert libraries[0]["source_type"] == "manual"
    assert libraries[0]["status"] == "ready"
