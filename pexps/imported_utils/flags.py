# MIT License

# Copyright (c) 2023 Vindula Jayawardana and Zhongxia Yan

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os

modules = ['numpy', 'pandas', 'scipy', 'matplotlib', 'torch', 'sklearn', 'visdom']
for name in modules:
    globals()[name] = name not in ('sklearn', 'visdom')

def none():
    for name in modules:
        globals()[name] = False

def include(*args):
    for name in args:
        assert name in modules
        globals()[name] = True

def exclude(*args):
    for name in args:
        assert name in modules
        globals()[name] = False

def only(*args):
    for name in args:
        assert name in modules
    for name in modules:
        globals()[name] = name in args