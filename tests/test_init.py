# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
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
