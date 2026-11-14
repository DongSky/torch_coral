"""Publication checks reject sensitive content without echoing its value."""
import importlib.util
from pathlib import Path

import pytest

_script = Path(__file__).resolve().parents[1] / 'scripts' / 'prepare-release.py'
if not _script.is_file():
    pytest.skip('Repository-only release helper is not bundled in distributions', allow_module_level=True)
_spec = importlib.util.spec_from_file_location('release_audit', _script)
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


@pytest.mark.parametrize('name', [
    '.env.production', 'CREDENTIALS.JSON', '.git-credentials', '.ssh/id_ed25519',
    'reports/result.json', '.GUI/settings.json', 'model.gguf', 'private.pem',
    'secrets.toml', '../outside.py',
])
def test_private_paths_are_rejected(name):
    with pytest.raises(ValueError):
        audit.inspect(name, b'innocent content')


@pytest.mark.parametrize('value', [
    'ghp_' + 'x' * 30,
    'person' + '@' + 'example.org',
    'C:' + '/Users/' + 'example/file.py',
    '/' + 'home/' + 'example/file.py',
    'https://' + 'person:password' + '@' + 'example.org',
])
def test_sensitive_content_is_redacted(value):
    with pytest.raises(ValueError) as error:
        audit.inspect('example.py', value.encode())
    assert value not in str(error.value)
    assert 'redacted' in str(error.value)


def test_portable_source_is_allowed():
    audit.inspect('torch_coral/example.py', b"path = root / 'models'\n")
