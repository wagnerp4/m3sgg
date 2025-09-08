# M3SGG Testing Framework

This directory contains the organized test suite for the M3SGG project, structured for efficient testing and CI/CD integration.

## Directory Structure

```
tests/
├── conftest.py                    # Shared pytest fixtures
├── test_config.py                 # Test configuration and utilities
├── unit/                          # Unit tests
│   ├── test_models/              # Model-specific tests
│   ├── test_datasets/            # Dataset tests
│   └── test_utils/               # Utility function tests
├── integration/                   # Integration tests
├── fixtures/                      # Test data and fixtures
│   ├── sample_videos/            # Sample video files
│   └── sample_annotations/       # Sample annotation files
└── performance/                   # Performance tests
```

## Running Tests

### Local Development

Use the provided PowerShell script for convenient test execution:

```powershell
# Run all tests
.\scripts\run_tests.ps1

# Run specific test types
.\scripts\run_tests.ps1 -TestType unit
.\scripts\run_tests.ps1 -TestType integration
.\scripts\run_tests.ps1 -TestType performance

# Run with coverage
.\scripts\run_tests.ps1 -TestType all -Coverage

# Run with verbose output
.\scripts\run_tests.ps1 -TestType unit -Verbose

# Run in parallel
.\scripts\run_tests.ps1 -TestType all -Parallel
```

### Direct pytest Commands

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/

# Run with coverage
pytest --cov=src/m3sgg --cov-report=html

# Run specific test files
pytest tests/unit/test_models/test_scenellm_components.py

# Run tests with specific markers
pytest -m "unit and not slow"
pytest -m "integration"
pytest -m "performance"
```

## Test Categories

### Unit Tests (`tests/unit/`)
- **Fast execution** (< 30 seconds per test)
- **Isolated testing** of individual components
- **No external dependencies** (databases, networks, etc.)
- **High coverage** of core functionality

### Integration Tests (`tests/integration/`)
- **Component interaction** testing
- **End-to-end workflows**
- **External service integration**
- **Configuration validation**

### Performance Tests (`tests/performance/`)
- **Resource-intensive** operations
- **Memory usage** monitoring
- **Execution time** benchmarks
- **Scalability** testing

## Test Markers

Tests are categorized using pytest markers:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.performance` - Performance tests
- `@pytest.mark.slow` - Tests taking > 30 seconds
- `@pytest.mark.gpu` - Tests requiring GPU
- `@pytest.mark.gui` - Tests requiring GUI
- `@pytest.mark.network` - Tests requiring network
- `@pytest.mark.data` - Tests requiring large data files

## CI/CD Integration

### GitHub Actions Workflow

The project includes a comprehensive GitHub Actions workflow (`.github/workflows/tests.yml`) that:

- **Runs on multiple Python versions** (3.10, 3.11, 3.12)
- **Tests on multiple operating systems** (Ubuntu, Windows, macOS)
- **Generates coverage reports** with HTML and XML output
- **Performs code quality checks** (flake8, black, isort, mypy)
- **Uploads test artifacts** for debugging
- **Comments coverage on pull requests**

### Workflow Triggers

- **Push to main/master/develop** branches
- **Pull requests** to main/master/develop branches
- **Manual dispatch** with custom parameters
- **Scheduled runs** (if configured)

## Test Configuration

### pytest.ini
Contains pytest configuration including:
- Test discovery patterns
- Default command-line options
- Timeout settings
- Warning filters
- Logging configuration

### conftest.py
Provides shared fixtures for:
- Test data paths
- Sample video/annotation files
- Temporary directories
- Common test utilities

### test_config.py
Centralized test configuration including:
- Path definitions
- Timeout settings
- Memory limits
- Environment setup

## Best Practices

### Writing Tests

1. **Use descriptive test names** that explain what is being tested
2. **Follow the AAA pattern**: Arrange, Act, Assert
3. **Use appropriate markers** for test categorization
4. **Keep tests independent** and isolated
5. **Use fixtures** for common setup/teardown
6. **Mock external dependencies** in unit tests

### Test Data

1. **Store test data** in `fixtures/` directory
2. **Use small, representative samples** for unit tests
3. **Include edge cases** and error conditions
4. **Clean up temporary files** after tests

### Performance Considerations

1. **Unit tests should be fast** (< 1 second each)
2. **Use parallel execution** for large test suites
3. **Skip slow tests** in development mode
4. **Monitor memory usage** in performance tests

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure `PYTHONPATH` includes project root
2. **Missing dependencies**: Install test dependencies with `pip install -e .[default]`
3. **Timeout errors**: Increase timeout in pytest.ini or use `--timeout` option
4. **Memory issues**: Use `--maxfail=1` to stop on first failure

### Debug Mode

Run tests with verbose output and debugging:

```bash
pytest -v -s --tb=long --pdb
```

### Coverage Analysis

Generate and view coverage reports:

```bash
pytest --cov=src/m3sgg --cov-report=html
open htmlcov/index.html
```
