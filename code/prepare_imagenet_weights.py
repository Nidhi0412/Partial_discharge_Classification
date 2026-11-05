"""Download and verify the official no-top weights used by installed Keras.

Uses only stdlib; no model is constructed and no data leaves this machine.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request


ROOT = Path(__file__).resolve().parents[1]


def constants(path):
    values = {}
    for node in ast.parse(path.read_text(encoding='utf-8')).body:
        if isinstance(node, ast.Assign):
            try: value = ast.literal_eval(node.value)
            except (ValueError, TypeError): continue
            for target in node.targets:
                if isinstance(target, ast.Name): values[target.id] = value
    return values


def md5(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'md5').hexdigest()


def prepare():
    source = Path(sys.prefix)/'Lib/site-packages/keras/src/applications'
    vgg = constants(source/'vgg16.py'); resnet = constants(source/'resnet.py')
    targets = [
        ('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', vgg['WEIGHTS_PATH_NO_TOP'],
         '6d6bbae143d832006294945121d1f1fc'),
        ('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
         resnet['BASE_WEIGHTS_PATH']+'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
         resnet['WEIGHTS_HASHES']['resnet50'][1])]
    cache = ROOT/'runtime_cache/keras/models'; cache.mkdir(parents=True, exist_ok=True)
    results = []
    for name, url, expected in targets:
        path = cache/name
        if not path.exists() or md5(path) != expected:
            temporary = path.with_suffix('.download')
            print(f'Downloading official ImageNet weights: {name}', flush=True)
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as stream:
                while chunk := response.read(1024*1024): stream.write(chunk)
            if md5(temporary) != expected:
                raise RuntimeError(f'Official weight hash mismatch: {name}')
            os.replace(temporary, path)
        results.append(dict(path=str(path), md5=expected, bytes=path.stat().st_size, origin=url))
        print(f'VERIFIED: {name}', flush=True)
    return results


if __name__ == '__main__':
    print(json.dumps(prepare(), indent=2))
