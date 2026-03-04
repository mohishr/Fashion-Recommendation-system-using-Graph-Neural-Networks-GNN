"""Preflight checks for the model_interfacing app.

This script does not require third-party modules.
It reports whether dependencies, checkpoints, and basic run command prerequisites are present.
"""

import importlib.util
import os
import sys

REQUIRED_MODULES = [
    'flask',
    'numpy',
    'tensorflow',
]
OPTIONAL_MODULES = [
    'pymongo',
]


def module_exists(name):
    return importlib.util.find_spec(name) is not None


def find_checkpoint_prefixes(weights_dir):
    if not os.path.isdir(weights_dir):
        return []

    prefixes = []
    for name in sorted(os.listdir(weights_dir)):
        if name.endswith('.index'):
            prefixes.append(os.path.join(weights_dir, name[:-6]))
    return prefixes


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.environ.get('NGNN_WEIGHTS_DIR', os.path.join(base_dir, 'model_weights'))

    print('Python:', sys.version.replace('\n', ' '))
    print('Base dir:', base_dir)
    print('Weights dir:', weights_dir)
    print('')

    missing = []
    print('[Required modules]')
    for mod in REQUIRED_MODULES:
        ok = module_exists(mod)
        print(' - {}: {}'.format(mod, 'OK' if ok else 'MISSING'))
        if not ok:
            missing.append(mod)

    print('\n[Optional modules]')
    for mod in OPTIONAL_MODULES:
        ok = module_exists(mod)
        print(' - {}: {}'.format(mod, 'OK' if ok else 'MISSING (feature limited)'))

    ckpts = find_checkpoint_prefixes(weights_dir)
    print('\n[Checkpoint files]')
    if ckpts:
        for path in ckpts:
            print(' - found:', path)
    else:
        print(' - none found (put .index/.meta/.data files under model_weights)')

    print('\n[Run command]')
    print(' cd {}'.format(base_dir))
    print(' NGNN_DISABLE_IMAGENET=1 python app.py')

    if missing:
        print('\nRESULT: FAIL (missing required modules: {})'.format(', '.join(missing)))
        sys.exit(1)

    if not ckpts:
        print('\nRESULT: WARN (app can start but will use random weights until checkpoint is provided)')
        sys.exit(2)

    print('\nRESULT: PASS (environment looks ready for local app launch)')


if __name__ == '__main__':
    main()
