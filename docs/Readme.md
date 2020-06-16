# Building the documentation

Install sphinx and dependencies:
```
pip install -r requirements.txt
```

Then build the html documentation:
```
make html
```

Run `make html` again whenever a change is made in the `source` folder. The
output html is generated in the `_build/html` folder. Open
`_build/html/index.html` in your browser to view the locally generated
documentation.

