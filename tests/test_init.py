# From: https://stackoverflow.com/questions/2481511/mocking-importerror-in-python
try:
    import builtins
except ImportError:
    import __builtin__ as builtins
realimport = builtins.__import__


def monkeypatched_import(name, *args):
    # raise ImportError
    if name == 'compressai.version':
        raise ImportError
    if name == 'range_coder':
        raise ImportError
    return realimport(name, *args)


builtins.__import__ = monkeypatched_import


def test_import_errors():
    # This should not crash
    import compressai


def test_version():
    builtins.__import__ = realimport
    from compressai.version import __version__
    assert len(__version__) == 5
