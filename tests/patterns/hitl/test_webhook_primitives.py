from __future__ import annotations

import hmac
import json
from hashlib import sha256
from unittest.mock import patch

import httpx
import pytest
from pydantic import ValidationError

from pydantic_ai_stateflow.patterns.hitl.channels.webhook import (
    WEBHOOK_SIGNATURE_HEADER,
    WebhookConfig,
    post_webhook,
    sign_payload,
)


def test_webhook_config_validates_url() -> None:
    cfg = WebhookConfig(url="https://example.com/cb", secret="sssh")
    assert str(cfg.url) == "https://example.com/cb"
    assert cfg.secret == "sssh"


def test_webhook_config_is_frozen() -> None:
    cfg = WebhookConfig(url="https://example.com/cb", secret="sssh")
    with pytest.raises(ValidationError):
        cfg.secret = "leaked"


def test_sign_payload_hmac_sha256_hex() -> None:
    body = b'{"hello":"world"}'
    sig = sign_payload(body, secret="sssh")
    expected = hmac.new(b"sssh", body, sha256).hexdigest()
    assert sig == expected
    assert all(c in "0123456789abcdef" for c in sig)
    assert len(sig) == 64


def test_sign_payload_different_secret_distinct() -> None:
    body = b"payload"
    assert sign_payload(body, secret="a") != sign_payload(body, secret="b")


def test_signature_header_name_is_canonical() -> None:
    assert WEBHOOK_SIGNATURE_HEADER == "X-Stateflow-Signature"


@pytest.mark.asyncio
async def test_post_webhook_sends_signature_and_body(
    fresh_dbos_executor: None,
) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *a: object) -> None:
            return None

        async def post(
            self, url: str, content: bytes, headers: dict[str, str]
        ) -> object:
            captured["url"] = url
            captured["content"] = content
            captured["headers"] = headers

            class _R:
                status_code = 200

                def raise_for_status(self) -> None:
                    return None

            return _R()

    with patch(
        "pydantic_ai_stateflow.patterns.hitl.channels.webhook.httpx.AsyncClient",
        FakeClient,
    ):
        body = json.dumps({"request_id": "abc"}).encode()
        await post_webhook(
            url="https://hooks.example/cb",
            body=body,
            signature="deadbeef",
        )
    assert captured["url"] == "https://hooks.example/cb"
    assert captured["content"] == body
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers["X-Stateflow-Signature"] == "deadbeef"
    assert headers["Content-Type"] == "application/json"


@pytest.mark.asyncio
async def test_post_webhook_raises_on_5xx(
    fresh_dbos_executor: None,
) -> None:
    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *a: object) -> None:
            return None

        async def post(
            self, url: str, content: bytes, headers: dict[str, str]
        ) -> object:
            class _R:
                status_code = 500

                def raise_for_status(self) -> None:
                    req = httpx.Request("POST", url)
                    raise httpx.HTTPStatusError(
                        "boom", request=req, response=httpx.Response(500)
                    )

            return _R()

    with patch(
        "pydantic_ai_stateflow.patterns.hitl.channels.webhook.httpx.AsyncClient",
        FakeClient,
    ), pytest.raises(httpx.HTTPStatusError):
        await post_webhook(url="https://x.example/cb", body=b"{}", signature="s")
