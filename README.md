# ADAPT: A Data Analysis Project Template

This is a lightweight and structured template for data analysis projects. The
goal is to keep work organized, reproducible, and easy to pick up or hand off.
We built this template with small to medium sized projects for medical data
analysis in mind, but serves other domains and larger projects as well.

Feedback and suggestions for improvement are welcome. Please [create an
issue]](https://github.com/ROTARC/ADAPT/issues) on GitHub.

### Note on this `README.md` file

Once you have created your own repository from this template, you should replace
this `README.md` file with your own. You can rename this file to `TEMPLATE.md`
or similar to keep it as a reference.

## Using this template

### Get the template files

#### Option 1: Create a repository from this template (recommended)

Follow the [instructions on
GitHub](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template)
to create your own repository from [the ADAPT
template](https://github.com/rotarc/adapt). Then, clone your own repository to
your computer to start working on it. There are several ways to clone a
repository, e.g., via [Visual Studio
Code](https://learn.microsoft.com/en-us/azure/developer/javascript/how-to/with-visual-studio-code/clone-github-repository?tabs=activity-bar),
[Git
Tower](https://www.git-tower.com/help/guides/manage-repositories/clone-remote-repository/windows)
or through [GitHub
itself](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository?tool=webui).

#### Option 2: Download the files

Download the latest version of this template of [the ADAPT
template](https://github.com/rotarc/adapt). Extract as your project directory.

### Set your project name

Before installing the project, set the project name. See
[pyproject.toml](#pyprojecttoml).

### Create a virtual environment and install your project

To create a virtual environment and install your project, run the commands below
in your terminal.

#### Option 1: `uv` (recommended)

1. Create a virtual environment

    ```sh
    uv venv --python 3.13
    ```

2. Install the current project.

    **NB: Before installing the project, set the project name. See [pyproject.toml](#pyprojecttoml).**
  
    ```sh
    uv sync
    ```

3. Run your code.

    ```sh
    uv run python analysis/example_script.py
    ```

#### Option 2: `conda`

We do no longer recommend `conda`. Only use `conda` if you use the `conda` or
`conda-forge` package repository, and you have the appropriate licenses to use
it.

1. Create a virtual environment.

   ```sh
   conda create --prefix "./.conda" python=3.12
   ```

   This template requires Python 3.11 or higher. At the time of writing, `conda`
   supports Python 3.12 or lower.

   Alternatively, you can create a named `conda` virtual environment. See [the
   documentation](https://docs.conda.io/projects/conda/en/latest/commands/create.html)
   for details.

2. Activate the virtual environment.

    ```sh
    conda activate ./.conda
    ```

3. Install project as editable package.

    **NB: Before installing the project, set the project name. See [pyproject.toml](#pyprojecttoml).**

    ```sh
    pip install -e .
    ```

## Directory structure

The directory structure of this template is designed to keep your project
organized and reproducible. Some important concepts are:

- Files with different purposes are stored in different directories.
- No extra files or directories should (generally) be stored in the root
  directory, with the exception of project metadata files (e.g., [`mkdocs.yml`](https://www.mkdocs.org/user-guide/configuration/)).
- Computer-readable outputs are considered data and should be stored in the
  `data/` directory, while human-readable outputs are stored in the `output/`
  directory.
- All configuration is done via TOML files in the `config/` directory; hard-coded
  values should be avoided.
- Reusable code should be stored in the `src/` directory, while scripts and
  notebooks should be stored in the `analysis/` directory.

Below is an overview of the main files/directories and their purpose:

```text
├── .vscode/                 # Editor configuration
├── analysis/                # Scripts and notebooks for analysis
├── config/                  # Configuration files (TOML format)
│   ├── .secrets.toml        # API keys, credentials (gitignored)
│   ├── settings.toml        # Base configuration
│   └── settings.local.toml  # User-specific overrides
├── data/ 
│   ├── annotations/         # Processing annotations
│   ├── interim/             # Intermediate processing results
│   ├── processed/           # Final data for analysis
│   └── source/              # Raw data from experiments or instruments
├── docs/                    # Documentation
├── output/ 
│   ├── figures/             # Saved plots and images
│   ├── reports/             # Generated reports (PDFs, etc.)
│   └── tables/              # Saved tables (CSV, Excel, etc.)
├── references/              # Papers, links, background materials
├── src/                     # Reusable Python modules
│   ├── __version__.py       # Project version
│   └── project
│       │   run.py           # Information and functions concerning the current project run
│       └── settings.py      # Code to load settings with interpolation
├── tests/                   # Tests for the project
├── .gitignore               # Git ignore file 
├── pyproject.toml           # Project metadata and dependencies
└── README.md                # This file
```

**Note**: Directories in this template that are otherwise empty contain a
`.gitkeep` file. These files only exist to store the directories in the
repository. You can remove these files after cloning your repository.

### Code

Script and notebook files should go in the `analysis/` directory. It is useful
to number them in order of logical operation. E.g.,

- `analysis/01-convert_data.py` converts source data to a more workable format
  in `data/interim/01-converted/`
- `analysis/02-filter_data.py` filters data and stores it in
  `data/interim/02-filtered/`
- `analysis/02a-remove_invalid_data.py` (was added after `03-...`)
- `analysis/03-select_features.py` automatically selects data to create the
  final version of the data in `data/processed/`
- `analysis/04-calculate_something.py` generates output data

Reusable code should go in the `src/` directory. As a rule: never import code
from a file in `analysis/`, but import code from modules in `src/`. We recommend
a workflow where you develop code as a script or notebook and — once finished —
formalize it in functions or classes in modules inside `src/`. This might also
be a good time to write documentation as well as tests for your code, which
should be stored in the `tests/` directory. Catherine Nelson wrote [a
book](https://www.amazon.com/Software-Engineering-Data-Scientists-Notebooks-ebook/dp/B0CWMCN8TD)
about this process and recently gave a talk about it at the [PyCon
2025](https://www.youtube.com/watch?v=o4hyA4hotxw) conference. There are many
other resources on this topic.

The package has been set up such that any module inside `src/` can be directly
imported. E.g.,

- `src/helper.py` can be imported using `import helper`
- `src/plotting/plot_dataset.py` can be imported using `from plotting import
plot_dataset.py` (or `import plotting` and using `plotting.plot_dataset(...)`).

### Data

All data should go in the `data/` directory. `data/source/` only contains source
data. Source data should be read-only. `data/annotations/` should contain extra
information about the source data. For example:

- `data/annotations/subjects.csv` is a list of subject numbers with data file
  names to be included in the analysis
- `data/annotations/selection/selection-{subject_id}.json` contains the time range to be
  selected from the data of the specific subject

### Output

Any output should go in the `output/` directory. It is useful to include in the
file name the number of the analysis file, then a number for subsequent outputs
from the same file, and a good description. E.g.,

- `outputs/figures/02-01-spectrogram-{subject_id}.png` for a spectrogram created
  in `02-filter_data.py`
- `outputs/figures/02a-02-filter-results-{subject_id}.png` for a plot of the
  filtered signal created in `02-filter_data.py`
- `outputs/figures/02a-01-removed_sections-{subject_id}.png` for a plot of
  removed sections from `02a-remove_invalid_data.py`
- `outputs/reports/04-01-feature-analysis-{subject_id}.pdf` for a PDF including
  figures and tables created by `04-calculate_something.py`

If there are many subjects or multiple figures per subject, it might be
convenient to put outputs in respective subdirectories. Always keep the full
filename, in case you collect them per subject.

- `outputs/figures/02-01-spectrogram/02-01-spectrogram-{subject_id}.png`

#### Run identifiers

It might be useful to add a unique identifier to the name of output files. This
prevents overwriting previous results and accidentally combining up-to-date and
out-of-date results. The identifier can be based on the current version of your
project (available as `project.version`), the current git commit hash,
[^gitpython] or a sequential run number. This template includes the function
`project.run.get_run_identifier()` that tracks the run number in a file (set by
`project.settings.paths.run_identifier`, default `output/.run_identifier`) and returns a
formatted version (format set as `project.settings.run_identifier.format`, default
`"run{:04d}"`, e.g., `"run0425"`).

### Documentation

Documentation and instructions about the analysis process should go in `docs/`.
This should at a minimum contain information on how to run the analysis. It
should also contain a description of how the source data and annotations should
be formatted. As a rule: if you give this project — minus the data — to someone
else, can they reproduce your analysis with their own dataset?

Reference files, e.g., scientific publications explaining your methods, should
be included in `references/`. Code comments or documentation should refer to
these references when appropriate.

## `pyproject.toml`

The pyproject file contains information about your project. You should give the
project a suitable name, description, and author(s) information. The version of
the project is stored in the file `src/__version__.py`.

Project dependencies, i.e., packages and libraries required by your code, should be
listed in `dependencies`. The only template dependency is
[OmegaConf](https://omegaconf.readthedocs.io/en/2.3_branch/) for configuration
management. After updating dependencies[^1], update your installation using `pip
install -e .` or `uv sync`.

[^1]: If you use `uv` you can add a dependency by running `uv add <package>` in
    a terminal instead of manually editing `pyproject.toml`.

Some info from the `pyproject.toml` is available in Python as `project.version`,
`project.name`. The complete file is available as a dictionary as
`project._pyproject_data`.

More information on `pyproject.toml` can be found in the [Python documentation](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/).

## Configuration

Settings should always be preferred over hard-coded values. This makes it easy
to find the parameters used to perform your analysis, but also prevents values
from being defined more than once, resulting in unexpected behavior.

This template offers an easy way to manage and load configuration settings. In
short, settings are stored in TOML files in the `config/` directory, and can be
accessed with:

```python
from project import load_settings
settings = load_settings()
```

All files with the `.toml` extension in the `config/` directory are
automatically detected and loaded in alphabetical order. Settings from later
files override those from earlier files, so you can control precedence by
naming.

The template only contains a single `config/settings.toml` file. In case of large
projects, you can split the configuration into multiple files. We recommend the
following structure:

- Number files to control the order in which they are loaded.
- Use a prefix to indicate the type of settings, e.g., `config/00-*` for path settings,
  `config/10-*` for filtering settings, etc.
- When further splitting files of the same type, increment the number,
  e.g., `config/01-paths-input.toml` for the first config file of a type,
  `config/02-paths-output.toml` for the second file, etc.
- Use `config/*.local.toml` for user-specific overrides, e.g., paths, that should not be
  shared with others. Files ending in `config/.local.toml` should not be committed to a
  git repository, and are therefore ignored using the `config/.gitignore` file.
- Files that start with a dot, e.g., `config/.secrets.toml`, are ignored by git and
  should not be committed to a git repository. These files can contain sensitive
  information, such as API keys or credentials.

### Example configration file structure

A possible configuration structure is:

```text
config/.secrets.toml          # API keys, credentials (gitignored)
config/00-paths.toml          # declares the base path
config/01-paths-input.toml    # declares inputs paths
config/02-paths-output.toml   # declares output paths
config/03-paths.local.toml    # declares user-specific overrides (gitignored)
config/10-filtering.toml
config/20-visualization.toml
```

### Interpolation

`${...}` syntax is supported for interpolation (via
[OmegaConf](https://omegaconf.readthedocs.io/en/2.3_branch/)). Interpolation is
re-evaluated after all configuration files have been loaded. This means that you
can overwrite settings used in earlier files in later files. For example, you
can set the base path in `config/01-paths-output.toml` to a different value than
in `config/00-paths-input.toml`.

### Local vs. shared config files

Shared files (e.g., `config/10-*.toml`) should contain all settings that are
shared between computers. However, settings that are specific to the computer
the code runs on (likely only the source path), should be in
`config/*.local.toml` or another appropriately named TOML file that is loaded
later alphabetically. If a local settings file is required, it is best to add an
example template file, e.g., `config/10-paths.local.toml.template` to the
repository.


**Example `config/00-settings.toml`**:

```toml
[paths.data]
base = "./data"
input = "${paths.data.base}/source"
output = "${paths.data.base}/processed"

[filtering]
method = "lowpass"
cuttof = 25  # Hz
order = 4
```

**Example `config/01-settings.local.toml`**:

```toml
[paths.data]
base = "/path/to/server"  # overrides setting in 00-settings.toml; paths.data.input and paths.data.output are also updated
```

Load configuration settings in Python:

```python
from project import settings

print(settings.filtering.order)
```

### Handling non-existing paths

All paths in the configuration are converted to `pathlib.Path` objects. If a
directory path does not exist, a warning message is printed. You can suppress
these warnings by passing `warn_if_missing=False` to `load_settings()`.

You can create all non-existing directories by running
`project.settings.create_missing_directories(settings)`, where `settings` is the
loaded settings object. This also creates any parent directories that do not
exist.

## Best practices

- Write a README.md (rename this file to TEMPLATE.md) with a short explanation
  of the repository: what is the goal, and how can people use it? For simple
  projects, this could suffice as documentation. For more complex projects, it
  should refer to more extensive documentation.
- All interim and processed data should be derived from source data and
  annotations. This derivation should be reproducible, such that interim and
  processed files can be removed without losing information.
- Files in `output/` should be human-readable. Data used for further analysis,
  e.g., a `pandas` data frame or export for further analysis in R, should be
  stored as interim data.
- Use git for source version control of your code. Regularly make atomic
  commits. Update the version in `src/__version__.py` and create a git tag at
  the corresponding commit whenever the code has reached a next milestone or
  stable state. Data (possibly except for test data) and output do not belong in
  a git repository.
- Analysis workflows often have two stages: first process and analysis data from
  individual measurements or subjects, then combine data for a final comparison.
  Avoid looping over measurements or subjects for complex analysis workflows in
  long script files. The best way to prevent this is by creating functions that
  do all the work. A script file then can loop over subjects and call the
  function for all subjects. However, this is not always a good option,
  especially in the early phases of analysis. A secondary option is to have a
  script file that performs analysis on a single subject.

  **Example `analysis/02-convert_data.py`**:

  ```python
  # 'subject_id' can be provided on the command line or by a calling script (see
  # below). If it was not provided, i.e., when the script is run without command
  # line arguments, the value is set to 'S004'.
  import argparse


  parser = argparse.ArgumentParser()
  parser.add_argument("--subject_id", type=str, default="S004")
  args, _ = parser.parse_known_args()
  subject_id = args.subject_id

  # <perform a lot of complicated steps>
  ```

  While developing this workflow, you can run this analysis for a single
  subject. Then, have a secondary file that runs `02-convert_data.py` for all
  subjects.

  **Example `data/annotations/subjects.csv`**:

  ```text
  S001,S003,S004,S005,S010
  ```

  **Example `analysis/02a-run-convert_data.py`**:

  ```python
  import csv
  import subprocess
  import sys
  from project_settings import settings

  with (settings.paths.data.annotations / "subjects.csv").open("r") as csvfile:
      # Get the ','-delimited values from the first line
      subjects = next(csv.reader(csvfile, delimiter=","))

  # Load script outside for loop to prevent re-reading (a different version of) the file
  script_file = "analysis/02-convert_data.py"

  for subject_id in subjects:
      command = [
          sys.executable,
          scriptfile,
          "--subject_id",
          subject_id,
      ]
      result = subprocess.run(command, text=True, capture_output=True)

      if result.returncode != 0:
          print(f"Error processing subject {subject_id}:", result.stderr)
          continue
  ```

[^gitpython]: You can get the commit hash using `gitpython`. Install `gitpython` using `pip`
    or `uv`.

    ```python
    from gitpython import Repo
    import os
    repo = Repo(os.getcwd())
    short_commit_sha = repo.head.commit.hexsha[:7]
    ```

    Commit hashes are not chronological if ordered by name. With a
    bit more work, you can add a commit count, resulting in a chronological order
    of files.

    ```python
    n_commits = len(tuple(repo.iter_commits()))  # Get the number of commits
    file_name = f"02-01-spectrogram-{subject_id}-{n_commits:03d}-{short_commit_sha}.png"
    ```

    Note that using git commit hashes requires frequent commits to be useful.
