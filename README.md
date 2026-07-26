# film-recommendations

[![Release](https://img.shields.io/github/v/release/tkm212/film-recommendations)](https://img.shields.io/github/v/release/tkm212/film-recommendations)
[![Build status](https://img.shields.io/github/actions/workflow/status/tkm212/film-recommendations/main.yml?branch=main)](https://github.com/tkm212/film-recommendations/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/tkm212/film-recommendations/branch/main/graph/badge.svg)](https://codecov.io/gh/tkm212/film-recommendations)
[![Commit activity](https://img.shields.io/github/commit-activity/m/tkm212/film-recommendations)](https://img.shields.io/github/commit-activity/m/tkm212/film-recommendations)
[![License](https://img.shields.io/github/license/tkm212/film-recommendations)](https://img.shields.io/github/license/tkm212/film-recommendations)

This is a repository to predict what films a user will like based off past ratings. This will leverage machine learning techniques.
This is currently a side-project so needs a lot of work to get into good shape!

- **Github repository**: <https://github.com/tkm212/film-recommendations/>
- **Documentation** <https://tkm212.github.io/film-recommendations/>

## Running the pipeline

A single CLI script trains the Ridge baseline model and serves a multi-page Dash dashboard at <http://127.0.0.1:8050> — an Overview page with cross-filterable metrics and charts, a Films explorer with posters and per-film details, and an About page describing the methodology:

```bash
# Serve the dashboard from the last saved predictions
uv run python scripts/main.py

# Retrain on the cached TMDB features (outputs/ratings_with_tmdb_features.csv), then serve
uv run python scripts/main.py --train

# Re-run the heavy TMDB feature engineering (live API calls, needs API_KEY in .env), train, then serve
uv run python scripts/main.py --ingest
```

Use `--port` to serve on a different port.

## Getting started with your project

### 1. Create a New Repository

First, create a repository on GitHub with the same name as this project, and then run the following commands:

```bash
git init -b main
git add .
git commit -m "init commit"
git remote add origin git@github.com:tkm212/film-recommendations.git
git push -u origin main
```

### 2. Set Up Your Development Environment

Then, install the environment and the pre-commit hooks with

```bash
make install
```

This will also generate your `uv.lock` file

### 3. Run the pre-commit hooks

Initially, the CI/CD pipeline might be failing due to formatting issues. To resolve those run:

```bash
uv run pre-commit run -a
```

### 4. Commit the changes

Lastly, commit the changes made by the two steps above to your repository.

```bash
git add .
git commit -m 'Fix formatting issues'
git push origin main
```

You are now ready to start development on your project!
The CI/CD pipeline will be triggered when you open a pull request, merge to main, or when you create a new release.

To finalize the set-up for publishing to PyPI, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/publishing/#set-up-for-pypi).
For activating the automatic documentation with MkDocs, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/mkdocs/#enabling-the-documentation-on-github).
To enable the code coverage reports, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/codecov/).

## Releasing a new version



---

Repository initiated with [fpgmaas/cookiecutter-uv](https://github.com/fpgmaas/cookiecutter-uv).
