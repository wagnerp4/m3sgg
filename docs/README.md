# DLHM VidSGG Documentation

This directory contains the Sphinx documentation for DLHM VidSGG.

## Building Documentation Locally

### Prerequisites

Install the required documentation dependencies:

```bash
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints myst-parser sphinx-copybutton sphinxcontrib-napoleon
```

### Building

#### On Linux/macOS

```bash
cd docs
make html
```

#### On Windows (PowerShell)

```powershell
cd docs
sphinx-build -b html . _build/html
```

#### Using Python directly

```bash
cd docs
python -m sphinx -b html . _build/html
```

### Viewing Documentation

After building, open `_build/html/index.html` in your browser.

### Available Make Targets

- `make html` - Build HTML documentation
- `make clean` - Clean build directory
- `make linkcheck` - Check for broken links
- `make livehtml` - Auto-rebuild on changes (requires sphinx-autobuild)
- `make strict` - Build with warnings as errors

## Documentation Structure

```
docs/
├── index.rst           # Main documentation page
├── installation.rst    # Installation guide
├── usage.rst          # Usage examples
├── datasets.rst       # Dataset documentation
├── models.rst         # Model documentation
├── training.rst       # Training guide
├── evaluation.rst     # Evaluation guide
├── api/               # API documentation
│   ├── dataloader.rst
│   ├── lib.rst
│   └── models.rst
├── contributing.rst   # Contribution guidelines
├── changelog.rst      # Change log
├── license.rst        # License information
├── conf.py           # Sphinx configuration
├── _static/          # Static assets (CSS, images)
└── _templates/       # Custom templates
```

## Writing Documentation

### reStructuredText (RST) Guidelines

- Use proper heading hierarchy (=, -, ~, ^)
- Include code blocks with appropriate syntax highlighting
- Use cross-references with `:doc:` and `:ref:`
- Add docstrings to all Python functions and classes

### API Documentation

API documentation is automatically generated from Python docstrings using sphinx.ext.autodoc. Follow these guidelines:

- Use Google or NumPy docstring format
- Include parameter types and descriptions
- Document return values and raised exceptions
- Add usage examples when helpful

Example:

```python
def example_function(param1: str, param2: int = 10) -> bool:
    """
    Brief description of the function.
    
    Args:
        param1: Description of the first parameter
        param2: Description of the second parameter (default: 10)
        
    Returns:
        Description of the return value
        
    Raises:
        ValueError: When param1 is empty
        
    Example:
        >>> result = example_function("test", 5)
        >>> print(result)
        True
    """
    return True
```

## Deployment

Documentation is automatically built and deployed to GitHub Pages when changes are pushed to the main branch. The deployment workflow is defined in `.github/workflows/docs.yml`.

### Manual Deployment

To deploy manually:

1. Build the documentation locally
2. Push the `_build/html` directory to the `gh-pages` branch
3. Enable GitHub Pages in repository settings

## Troubleshooting

### Common Issues

**Import Errors During Build**
- Ensure all project dependencies are installed
- Add missing packages to the documentation requirements

**Warnings About Missing References**
- Check that all `:doc:` and `:ref:` links are valid
- Verify file names and paths are correct

**Theme Issues**
- Ensure sphinx-rtd-theme is installed
- Check theme configuration in conf.py

**Broken Links**
- Run `make linkcheck` to identify broken external links
- Update or remove broken references

### Getting Help

If you encounter issues:

1. Check the Sphinx documentation: https://www.sphinx-doc.org/
2. Review the build output for specific error messages
3. Create an issue in the repository with details about the problem
