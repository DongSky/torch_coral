"""Check public names only. Never authenticates, reserves a name or uploads."""
import argparse
import datetime
import json
from pathlib import Path
import re
import urllib.error
import urllib.request


def check(names):
    records = []
    for name in names:
        canonical = re.sub(r'[-_.]+', '-', name).lower()
        record = {'requested': name, 'normalized': canonical, 'checks': []}
        for url in (f'https://pypi.org/pypi/{name}/json', f'https://pypi.org/simple/{canonical}/'):
            try:
                request = urllib.request.Request(url, headers={'User-Agent': 'torch-coral-release-name-check/0.1'})
                with urllib.request.urlopen(request, timeout=30) as response:
                    status = response.status
            except urllib.error.HTTPError as exc:
                status = exc.code
            record['checks'].append({'url': url, 'http_status': status})
        record['visible_project_absent'] = all(r['http_status'] == 404 for r in record['checks'])
        records.append(record)
    return {'checked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'results': records,
            'limitation': '404 means no publicly visible project. It does not reserve the name or guarantee registration; recheck immediately before upload.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='reports/pypi-name-check.json')
    args = parser.parse_args()
    result = check(['torch-coral', 'torch_coral', 'torch.coral', 'torch-coral-local', 'coral-torch'])
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
