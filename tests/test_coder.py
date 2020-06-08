import pytest

import compressai


def test_get_entropy_coder():
    assert compressai.get_entropy_coder() == 'ans'


def test_available_entropy_coders():
    rv = compressai.available_entropy_coders()

    assert isinstance(rv, list)
    assert 'ans' in rv


def test_set_entropy_coder():
    compressai.set_entropy_coder('ans')

    with pytest.raises(ValueError):
        compressai.set_entropy_coder('cabac')
