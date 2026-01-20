# Contributing Guide

Thank you for your interest in contributing to the Indonesia Economic Forecasting System! This document provides guidelines and instructions for contributing.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Code Standards](#code-standards)
- [Pull Request Process](#pull-request-process)
- [Testing](#testing)
- [Documentation](#documentation)

---

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the project, not personal opinions
- Accept responsibility for mistakes

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or insulting comments
- Publishing private information
- Unprofessional conduct

---

## Getting Started

### Prerequisites

- Python 3.9+
- Git
- GitHub account

### Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/Predictive-Analytics-for-Indonesia-s-Economic-Forecasting.git
cd Predictive-Analytics-for-Indonesia-s-Economic-Forecasting

# Add upstream remote
git remote add upstream https://github.com/alazkiyai09/Predictive-Analytics-for-Indonesia-s-Economic-Forecasting.git
```

---

## Development Setup

### 1. Create Virtual Environment

```bash
cd refactored
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 2. Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Development dependencies
pip install pytest pytest-cov black flake8 mypy isort
```

### 3. Verify Setup

```bash
# Run tests
pytest tests/ -v

# Check formatting
black --check .

# Check linting
flake8 .
```

---

## How to Contribute

### Types of Contributions

| Type | Description | Label |
|------|-------------|-------|
| **Bug Fix** | Fix existing issues | `bug` |
| **Feature** | Add new functionality | `enhancement` |
| **Documentation** | Improve docs | `documentation` |
| **Tests** | Add/improve tests | `testing` |
| **Refactor** | Code improvements | `refactor` |

### Contribution Workflow

1. **Check existing issues** - See if someone is already working on it
2. **Create an issue** - Describe what you want to do
3. **Fork the repository**
4. **Create a feature branch**
5. **Make your changes**
6. **Write/update tests**
7. **Submit a pull request**

### Creating Issues

**Bug Report Template:**
```markdown
## Bug Description
Clear description of the bug.

## Steps to Reproduce
1. Step 1
2. Step 2
3. ...

## Expected Behavior
What should happen.

## Actual Behavior
What actually happens.

## Environment
- OS: Windows/Mac/Linux
- Python version: 3.x
- Package versions: (from pip freeze)
```

**Feature Request Template:**
```markdown
## Feature Description
What feature would you like?

## Use Case
Why is this needed?

## Proposed Solution
How should it work?

## Alternatives Considered
Other options you've considered.
```

---

## Code Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

```python
# Good: Clear, descriptive names
def calculate_moving_average(prices: pd.Series, window: int) -> pd.Series:
    """Calculate simple moving average."""
    return prices.rolling(window).mean()

# Bad: Unclear names
def calc(p, w):
    return p.rolling(w).mean()
```

### Required for All Code

1. **Type Hints**
```python
def train_model(
    features: pd.DataFrame,
    target: pd.Series,
    model_type: str = "lstm"
) -> Tuple[ModelTrainer, Dict[str, float]]:
```

2. **Docstrings** (Google style)
```python
def forecast(self, n_steps: int) -> np.ndarray:
    """
    Generate forecast.

    Args:
        n_steps: Number of steps to forecast.

    Returns:
        Array of forecast values.

    Raises:
        ValueError: If model not trained.
    """
```

3. **Logging** (not print)
```python
# Good
logger.info(f"Loaded {len(df)} rows")

# Bad
print(f"Loaded {len(df)} rows")
```

### Code Formatting

```bash
# Format with Black (line length 88)
black .

# Sort imports
isort .

# Check style
flake8 .
```

### Pre-commit Hook (Recommended)

Create `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.0.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
```

Install:
```bash
pip install pre-commit
pre-commit install
```

---

## Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Commit messages are clear

### Branch Naming

```
feature/add-arima-model
bugfix/fix-memory-leak
docs/update-api-reference
refactor/simplify-preprocessing
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add ARIMA model support
fix: resolve memory leak in data loader
docs: update API reference for forecaster
test: add unit tests for preprocessing
refactor: simplify feature engineering pipeline
```

### Pull Request Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
How was this tested?

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-reviewed code
- [ ] Added comments for complex logic
- [ ] Updated documentation
- [ ] Added tests
- [ ] All tests pass
```

### Review Process

1. Create PR with clear description
2. Wait for automated checks
3. Address review comments
4. Get approval from maintainer
5. Squash and merge

---

## Testing

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_preprocessing.py -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

### Writing Tests

```python
# tests/test_my_feature.py
import pytest
import numpy as np
import pandas as pd

class TestMyFeature:
    """Tests for my_feature module."""

    def test_basic_functionality(self):
        """Test basic case."""
        result = my_function(input_data)
        assert result == expected_output

    def test_edge_case(self):
        """Test edge case."""
        result = my_function([])
        assert result == []

    def test_error_handling(self):
        """Test error is raised for invalid input."""
        with pytest.raises(ValueError):
            my_function(invalid_input)

    @pytest.fixture
    def sample_data(self):
        """Create sample data for tests."""
        return pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=100),
            'Value': np.random.randn(100)
        })

    def test_with_fixture(self, sample_data):
        """Test using fixture."""
        result = process_data(sample_data)
        assert len(result) == 100
```

### Test Coverage Requirements

- Minimum 80% coverage for new code
- All public functions must have tests
- Edge cases should be covered

---

## Documentation

### Updating Documentation

1. **Docstrings** - Update in code
2. **README.md** - Main documentation
3. **API_REFERENCE.md** - API documentation
4. **QUICK_START.md** - Getting started guide

### Documentation Style

```python
def forecast(
    self,
    features: np.ndarray,
    n_steps: int = 12,
    return_confidence: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Generate economic forecast.

    This method uses the trained model to generate forecasts for the
    specified number of future time steps.

    Args:
        features: Input features array of shape (lookback, n_features).
            Should contain the most recent historical data.
        n_steps: Number of steps to forecast. Default is 12 (monthly).
        return_confidence: If True, return confidence intervals.

    Returns:
        If return_confidence is False:
            np.ndarray: Forecast values of shape (n_steps,)
        If return_confidence is True:
            Tuple of (forecast, lower_bound, upper_bound)

    Raises:
        ValueError: If model has not been trained.
        ValueError: If features shape is invalid.

    Example:
        >>> forecaster = EconomicForecaster(trainer=trained_model)
        >>> forecast = forecaster.forecast(features, n_steps=12)
        >>> print(forecast.shape)
        (12,)

    Note:
        Confidence intervals are calculated using historical error
        distribution and increase with forecast horizon.
    """
```

---

## Project Structure for New Features

When adding a new feature:

```
refactored/
├── new_module/
│   ├── __init__.py      # Export public interface
│   └── implementation.py # Main implementation
├── tests/
│   └── test_new_module.py # Tests
└── docs/
    └── NEW_MODULE.md     # Documentation (if significant)
```

### Example: Adding a New Model

1. **Create model** in `models/`
```python
# models/my_model.py
class MyModel:
    def __init__(self, params):
        ...
    def fit(self, X, y):
        ...
    def predict(self, X):
        ...
```

2. **Update factory** in `models/architectures.py`
```python
# Add to ModelFactory.AVAILABLE_MODELS
AVAILABLE_MODELS = ['lstm', 'gru', 'cnn_lstm', 'my_model']

# Add builder
def build_my_model(...):
    ...
```

3. **Add tests** in `tests/`
```python
# tests/test_my_model.py
def test_my_model_training():
    ...
```

4. **Update documentation**
- Add to README.md models section
- Add to API_REFERENCE.md

---

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open an Issue
- **Security**: Email maintainer directly

---

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing!**
