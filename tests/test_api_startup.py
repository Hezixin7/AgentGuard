import importlib
import json
import sys

import pytest

from deployment.config import (
    StartupConfigurationError,
    build_app_from_env,
    load_api_keys,
)
from deployment.deepseek_adapter import DeepSeekModelAdapter


def _env():
    return {
        "DEEPSEEK_API_KEY": "test-deepseek-key",
        "AGENTGUARD_API_KEYS": json.dumps(
            {
                "test-bearer-token": "user-1",
                "other-bearer-token": "user-2",
            }
        ),
    }


def test_load_api_keys_parses_a_server_side_mapping():
    keys = load_api_keys(
        '{"token-a":"user-1","token-b":"user-2"}'
    )
    assert keys == {"token-a": "user-1", "token-b": "user-2"}


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "   ",
        "not json",
        "[]",
        "{}",
        '{"token": 7}',
        '{"": "user-1"}',
        '{"token": ""}',
        '{"token":"user-1","token":"user-2"}',
    ],
)
def test_missing_or_malformed_api_keys_are_rejected(raw):
    with pytest.raises(StartupConfigurationError):
        load_api_keys(raw)


def test_api_key_errors_do_not_echo_secret_material():
    secret = "super-secret-bearer-token"
    for raw in (
        '["super-secret-bearer-token"]',
        '{"super-secret-bearer-token":',
        '{"super-secret-bearer-token": 7}',
        '{"super-secret-bearer-token":"user-1",'
        '"super-secret-bearer-token":"user-2"}',
    ):
        with pytest.raises(StartupConfigurationError) as error:
            load_api_keys(raw)
        assert secret not in str(error.value)


def test_missing_deepseek_key_is_rejected():
    env = {
        "AGENTGUARD_API_KEYS": '{"token-a":"user-1"}',
    }
    with pytest.raises(StartupConfigurationError) as error:
        build_app_from_env(env)
    assert "DEEPSEEK_API_KEY" in str(error.value)
    assert "token-a" not in str(error.value)


def test_valid_environment_builds_the_deepseek_app():
    app = build_app_from_env(_env())
    assert app is not None
    service = app.state.service
    assert isinstance(
        service.orchestrator._model,
        DeepSeekModelAdapter,
    )
    principal = app.state.authenticator.authenticate(
        "Bearer test-bearer-token"
    )
    assert principal.user_id == "user-1"
    routes = {route.path for route in app.routes}
    assert "/v1/runs" in routes
    assert "/v1/runs/{run_id}/confirmation" in routes
    assert "/v1/runs/{run_id}/trace" in routes
    assert "/healthz" in routes


def test_main_entrypoint_can_be_imported_with_valid_environment(
    monkeypatch,
):
    for key, value in _env().items():
        monkeypatch.setenv(key, value)
    sys.modules.pop("deployment.main", None)
    try:
        module = importlib.import_module("deployment.main")
        assert module.app is not None
    finally:
        sys.modules.pop("deployment.main", None)


def test_main_entrypoint_fails_startup_without_configuration(
    monkeypatch,
):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("AGENTGUARD_API_KEYS", raising=False)
    sys.modules.pop("deployment.main", None)
    try:
        with pytest.raises(StartupConfigurationError) as error:
            importlib.import_module("deployment.main")
        message = str(error.value)
        assert "AGENTGUARD_API_KEYS" in message
        assert "DEEPSEEK_API_KEY" not in message
    finally:
        sys.modules.pop("deployment.main", None)
