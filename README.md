# Setup enviroment

```bash
cd path/to/repository/root
export ROOT="$(pwd)"
pipenv shell
pipenv install
```

# launch ml

```bash
export PYTHONPATH="${PYTHONPATH}:${ROOT}/src"
python src/main.py
```

check out grid_search_results.pickle file in you work directory
