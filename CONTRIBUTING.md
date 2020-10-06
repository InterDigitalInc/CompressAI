# Contributing

If you want to contribute bug-fixes please directly file a pull-request. If you
plan to introduce new features or extend CompressAI, please first open an issue
to start a public discussion or contact us directly.

## Coding style

We try to follow PEP 8 recommendations. Automatic formatting is performed via
[black](https://github.com/google/yapf://github.com/psf/black) and
[isort](https://github.com/timothycrosley/isort/).

## Testing

We use [pytest](https://docs.pytest.org/en/5.4.3/getting-started.html). To run
all the tests:

* `pip install pytest pytest-cov coverage`
* `python -m pytest --cov=compressai -s`
* You can run `coverage report` or `coverage html` to visualize the tests
  coverage analysis

## Documentation

See `docs/Readme.md` for more information.

## Licence

By contributing to CompressAI, you agree that your contributions will be
licensed under the same license as described in the LICENSE file at the root of
this repository.

