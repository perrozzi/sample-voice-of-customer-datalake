"""One retry policy for every Bedrock caller, and the guard that keeps it that way.

The policy is: **retry a throttle, never retry a read timeout.** botocore cannot
express it — its attempt budget covers both — which is why `shared/aws.py` gives
the shared client ONE attempt and `shared/converse.py::bedrock_call_with_retry`
owns the decision instead.

That split has a failure mode this file exists to prevent. A caller that builds its
own Converse/InvokeModel request (tool config, image block, Anthropic-native body)
and calls the shared client directly gets NO retry at all, so the first 429 becomes
a failed job or a user-visible error. Three such callers exist, and nothing about
the client's signature makes the omission visible at the call site — the code reads
perfectly fine either way.
"""
import ast
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError, ReadTimeoutError

# lambda/shared/test/ -> voc-datalake/lambda/
LAMBDA_ROOT = Path(__file__).resolve().parents[2]

RAW_CALL_ATTRS = frozenset({'converse', 'invoke_model'})

# Modules that reach Bedrock without the shared policy, each with the reason it is
# allowed to. An empty exemption is not an option: the point of the list is that
# adding a caller forces a decision, in review, about what its first throttle does.
EXEMPT = {
    # Owns the policy.
    'shared/converse.py': 'defines bedrock_call_with_retry',
    # Degrades to a template prompt on any failure (see its own except branch), so a
    # throttle costs prompt quality rather than the avatar. Its IMAGE client is a
    # separate one that keeps botocore retries on purpose, because a 10-wide avatar
    # fan-out makes throttling the expected failure there. Also deliberately keeps a
    # narrow import graph (test_avatar asserts importing it pulls in no cryptography),
    # which importing converse would widen.
    'shared/avatar.py': 'falls back to a template prompt; image client retries separately',
    # stdlib+boto3 only so CoreStack stays container-free, so it CANNOT import
    # shared/. Builds its own client, already at one attempt, and its caller marks
    # the record failed with a stall guard behind it.
    'product_doc_extractor/handler.py': 'cannot import shared/ by design',
}


def _first_party_modules():
    """Every non-test .py under lambda/, excluding vendored layer code."""
    for path in sorted(LAMBDA_ROOT.rglob('*.py')):
        rel = path.relative_to(LAMBDA_ROOT).as_posix()
        if rel.startswith(('layers/', 'test')) or '/test' in f'/{rel}':
            continue
        yield rel, path


class TestEveryRawBedrockCallCarriesThePolicy:
    """A source-level guard, because there is no runtime signal to assert on.

    Aimed at the AST rather than a grep: a call is judged by whether a
    `bedrock_call_with_retry(...)` encloses it, which is exactly the property that
    makes the retry apply, and which no amount of nearby text can fake.
    """

    @staticmethod
    def _unguarded_calls(source: str) -> list[str]:
        tree = ast.parse(source)
        guarded: set[int] = set()

        # Mark every node inside a bedrock_call_with_retry(...) call, at any depth:
        # the real call sites pass a lambda, so the client call is a grandchild.
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, 'attr', None)
            if name == 'bedrock_call_with_retry':
                guarded.update(id(child) for child in ast.walk(node))

        return [
            f'{node.func.attr}() at line {node.lineno}'
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in RAW_CALL_ATTRS
            and id(node) not in guarded
        ]

    def test_no_module_calls_the_shared_client_without_a_retry(self):
        offenders: dict[str, list[str]] = {}

        for rel, path in _first_party_modules():
            source = path.read_text(encoding='utf-8')
            if 'get_bedrock_client' not in source and 'bedrock-runtime' not in source:
                continue
            if rel in EXEMPT:
                continue
            unguarded = self._unguarded_calls(source)
            if unguarded:
                offenders[rel] = unguarded

        assert offenders == {}, (
            f'these Bedrock calls have no retry: {offenders}. The shared client makes '
            f'ONE botocore attempt by design (BEDROCK_READ_TIMEOUT_SECONDS in '
            f'shared/aws.py), so a bare call surfaces the first ThrottlingException to '
            f'the user or fails the job. Wrap it in bedrock_call_with_retry(), use '
            f'converse(), or add the module to EXEMPT above with the reason.'
        )

    def test_the_exemptions_still_exist(self):
        """An exemption for a moved file silently stops guarding its replacement."""
        for rel in EXEMPT:
            assert (LAMBDA_ROOT / rel).is_file(), (
                f'{rel} is exempt from the Bedrock retry guard but no longer exists — '
                f'delete the exemption, or point it at the file that replaced it'
            )

    def test_the_guard_sees_an_unguarded_call(self):
        """The detector, on a mutant. Without this the case above can pass vacuously."""
        guarded = 'bedrock_call_with_retry(lambda: client.converse(modelId=m))'
        bare = 'client.converse(modelId=m)'

        assert self._unguarded_calls(guarded) == []
        assert self._unguarded_calls(bare) == ['converse() at line 1']


class TestTheRetryPolicy:
    """The two halves of the policy, in both directions."""

    @staticmethod
    def _throttle():
        return ClientError(
            {'Error': {'Code': 'ThrottlingException', 'Message': 'Rate exceeded'}},
            'Converse',
        )

    @patch('shared.converse.time.sleep')
    def test_a_throttle_is_retried_with_backoff(self, mock_sleep):
        """The half botocore used to provide for the raw callers, now provided here."""
        from shared.converse import bedrock_call_with_retry

        call = MagicMock(side_effect=[self._throttle(), {'ok': True}])
        result = bedrock_call_with_retry(call, step_name='t')

        assert result == {'ok': True}
        assert call.call_count == 2
        mock_sleep.assert_called_once()

    @patch('shared.converse.time.sleep')
    def test_a_read_timeout_is_not_retried(self, mock_sleep):
        from shared.converse import bedrock_call_with_retry

        call = MagicMock(side_effect=ReadTimeoutError(endpoint_url='https://bedrock'))

        with pytest.raises(ReadTimeoutError):
            bedrock_call_with_retry(call, step_name='t')

        assert call.call_count == 1
        mock_sleep.assert_not_called()

    @patch('shared.converse.time.sleep')
    def test_sustained_throttling_raises_a_named_error(self, mock_sleep):
        from shared.converse import BedrockThrottlingError, bedrock_call_with_retry

        call = MagicMock(side_effect=self._throttle())

        with pytest.raises(BedrockThrottlingError):
            bedrock_call_with_retry(call, max_retries=3, step_name='t')

        assert call.call_count == 3

    @patch('shared.converse.time.sleep')
    def test_sustained_throttling_returns_none_when_asked_not_to_raise(self, mock_sleep):
        """`None`, not `{}`: the helper is generic, so it cannot invent an empty
        response shape. `_converse_with_retry` maps it back to `{}` for its own
        callers, who are written against that."""
        from shared.converse import bedrock_call_with_retry

        call = MagicMock(side_effect=self._throttle())
        result = bedrock_call_with_retry(
            call, max_retries=2, raise_on_throttle=False, step_name='t'
        )

        assert result is None

    def test_a_non_retryable_error_is_raised_at_once(self):
        from shared.converse import bedrock_call_with_retry

        denied = ClientError(
            {'Error': {'Code': 'AccessDeniedException', 'Message': 'no'}}, 'Converse'
        )
        call = MagicMock(side_effect=denied)

        with pytest.raises(ClientError):
            bedrock_call_with_retry(call, step_name='t')

        assert call.call_count == 1
