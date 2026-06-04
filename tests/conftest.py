"""Shared pytest fixtures for all test modules."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine
from sqlalchemy.pool import StaticPool

import app.database as db_module
from app import main as app_module


@pytest.fixture(name="test_engine")
def test_engine_fixture():
    """
    Isolated SQLite engine using StaticPool so all connections share the
    SAME in-memory database. Critical for tests that seed data in one
    Session and read it via the API in a different Session.
    """
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    db_module.set_engine(engine)
    yield engine
    db_module.clear_engine()
    SQLModel.metadata.drop_all(engine)


@pytest.fixture(name="client")
def client_fixture(test_engine):
    """FastAPI test client. test_engine fixture runs first (sets engine override)."""
    with TestClient(app_module.app, raise_server_exceptions=True) as client:
        yield client
