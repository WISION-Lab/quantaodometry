"""
Tasks for maintaining the project.

Execute 'invoke --collection=dev --list' for guidance on using Invoke
"""
import fnmatch
import glob
import os
import platform
import shutil
import webbrowser
from pathlib import Path

from invoke import task

MAX_LINE_LENGTH = 121

ROOT_DIR = Path(__file__).parent
SETUP_FILE = ROOT_DIR.joinpath("setup.py")
TEST_DIR = ROOT_DIR.joinpath("tests")
SOURCE_DIR = ROOT_DIR.joinpath("src")
COVERAGE_FILE = ROOT_DIR.joinpath(".coverage")
COVERAGE_DIR = ROOT_DIR.joinpath("htmlcov")
COVERAGE_REPORT = COVERAGE_DIR.joinpath("index.html")
PYTHON_DIRS = [str(d) for d in [SOURCE_DIR, TEST_DIR]]


def _delete_file(file, except_patterns=None):
    if os.path.isfile(file):
        print(f"Removing file {file}...")
        os.remove(file)
    elif os.path.isdir(file):
        if except_patterns is None:
            print(f"Removing directory {file}...")
            shutil.rmtree(file, ignore_errors=True)
        else:
            print(f"Purging directory {file}...")
            for dirpath, dirnames, filenames in os.walk(file):
                # Remove regular files, ignore directories
                for filename in filenames:
                    file = os.path.join(dirpath, filename)
                    if any(fnmatch.fnmatch(file, pattern) for pattern in except_patterns):
                        print(f"\tKeeping file {file}...")
                    else:
                        print(f"\tRemoving file {file}...")
                        os.remove(file)
                # Remove empty directories
                if not dirnames and not filenames:
                    print(f"\tRemoving directory {file}...")
                    shutil.rmtree(dirpath, ignore_errors=True)


def _delete_pattern(pattern):
    for file in glob.glob(os.path.join("**", pattern), recursive=True):
        _delete_file(file)


def _run(c, command):
    return c.run(command, pty=platform.system() != "Windows")


@task(help={"check": "Checks if source is formatted without applying changes"})
def format(c, check=False):
    """Format code"""
    python_dirs_string = " ".join(PYTHON_DIRS + glob.glob(os.path.join(ROOT_DIR, "*.py")))
    # Run isort
    isort_options = f"--line-length={MAX_LINE_LENGTH}" + " --check-only --diff" if check else ""
    _run(c, f"isort {isort_options} {python_dirs_string}")
    # Run Black
    yapf_options = "--diff --check" if check else ""
    _run(c, f"black --line-length={MAX_LINE_LENGTH} {yapf_options} {python_dirs_string}")


@task
def lint_flake8(c):
    """Lint code with flake8"""
    _run(c, f"flake8 {' '.join(PYTHON_DIRS)} --max-line-length={MAX_LINE_LENGTH}")


@task
def lint_pylint(c):
    """Lint code with pylint"""
    _run(c, f"pylint --max-line-length={MAX_LINE_LENGTH} {' '.join(PYTHON_DIRS)}")


@task(lint_flake8, lint_pylint)
def lint(c):
    """Run all linting"""


@task
def test(c):
    """Run tests"""
    _run(c, "pytest")


@task
def precommit(c):
    """Run pre-commit hooks"""
    _run(c, "pre-commit run --all-files")


@task
def coverage(c):
    """Create coverage report"""
    _run(c, f"coverage run --source {SOURCE_DIR} -m pytest")
    _run(c, "coverage report")

    # Build a local report
    _run(c, "coverage html")
    webbrowser.open(COVERAGE_REPORT.as_uri())


@task
def clean_build(c):
    """Clean up files from package building"""
    _delete_file("build/")
    _delete_file("dist/")
    _delete_file(".eggs/")
    _delete_pattern("*.egg-info")
    _delete_pattern("*.egg")


@task
def clean_python(c):
    """Clean up python file artifacts"""
    _delete_pattern("__pycache__")
    _delete_pattern("*.pyc")
    _delete_pattern("*.pyo")
    _delete_pattern("*~")


@task
def clean_tests(c):
    """Clean up files from testing"""
    _delete_file(COVERAGE_FILE)
    _delete_file(COVERAGE_DIR)
    _delete_pattern(".pytest_cache")


@task(pre=[clean_build, clean_python, clean_tests])
def clean(c):
    """Runs all clean sub-tasks"""
    _delete_file("hparams.yaml")
