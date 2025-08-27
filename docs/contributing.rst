Contributing
============

We welcome contributions to DLHM VidSGG! This guide explains how to contribute to the project.

Getting Started
---------------

Development Setup
~~~~~~~~~~~~~~~~~

1. Fork the repository on GitHub
2. Clone your fork locally:

.. code-block:: bash

   git clone https://github.com/your-username/DLHM_VidSGG.git
   cd DLHM_VidSGG

3. Create a development environment:

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"

4. Install pre-commit hooks:

.. code-block:: bash

   pre-commit install

Code Style
~~~~~~~~~~

We follow Python PEP 8 style guidelines with some project-specific conventions:

* Use double quotes for strings
* Line length limit: 100 characters
* Use type hints for function parameters and return values
* Add docstrings to all public functions and classes

.. code-block:: python

   def example_function(param1: str, param2: int) -> bool:
       """
       Example function with proper style.
       
       Args:
           param1: Description of first parameter
           param2: Description of second parameter
           
       Returns:
           Description of return value
       """
       return True

Types of Contributions
----------------------

Bug Reports
~~~~~~~~~~~

When reporting bugs, please include:

* Python version and operating system
* Full error traceback
* Minimal code example to reproduce the issue
* Expected vs actual behavior

Use the bug report template:

.. code-block:: text

   **Bug Description**
   A clear description of the bug.
   
   **Steps to Reproduce**
   1. First step
   2. Second step
   3. Error occurs
   
   **Expected Behavior**
   What should happen.
   
   **Environment**
   - OS: [e.g., Ubuntu 20.04]
   - Python: [e.g., 3.10.0]
   - PyTorch: [e.g., 1.12.0]

Feature Requests
~~~~~~~~~~~~~~~~

For feature requests, please:

* Check if the feature already exists
* Describe the motivation and use case
* Provide a detailed specification
* Consider implementation complexity

New Models
~~~~~~~~~~

To contribute a new model:

1. Implement the model following the base model interface
2. Add comprehensive tests
3. Include training and evaluation scripts
4. Provide documentation and examples
5. Compare against existing baselines

.. code-block:: python

   from lib.base_model import BaseModel
   
   class NewModel(BaseModel):
       """New model implementation."""
       
       def __init__(self, config):
           super().__init__(config)
           # Model implementation
           
       def forward(self, inputs):
           # Forward pass implementation
           pass

Dataset Support
~~~~~~~~~~~~~~~

To add support for a new dataset:

1. Create a dataloader following the base dataset interface
2. Implement proper preprocessing and augmentation
3. Add dataset documentation
4. Provide download and setup instructions

.. code-block:: python

   from dataloader.base import BaseDataset
   
   class NewDataset(BaseDataset):
       """New dataset implementation."""
       
       def __init__(self, data_path, split, mode):
           super().__init__(data_path, split, mode)
           # Dataset initialization
           
       def __getitem__(self, idx):
           # Data loading implementation
           pass

Development Workflow
--------------------

Branch Naming
~~~~~~~~~~~~~

Use descriptive branch names:

* ``feature/model-name`` - for new models
* ``bugfix/issue-description`` - for bug fixes
* ``docs/section-name`` - for documentation updates
* ``refactor/component-name`` - for code refactoring

Commit Messages
~~~~~~~~~~~~~~~

Follow conventional commit format:

.. code-block:: text

   type(scope): description
   
   [optional body]
   
   [optional footer]

Examples:

.. code-block:: text

   feat(models): add Tempura model implementation
   
   fix(dataloader): resolve Action Genome loading issue
   
   docs(api): update model documentation
   
   test(evaluation): add unit tests for recall metrics

Pull Request Process
~~~~~~~~~~~~~~~~~~~~

1. Create a feature branch from main
2. Make your changes with appropriate tests
3. Update documentation if needed
4. Ensure all tests pass
5. Submit a pull request with clear description

Pull Request Template:

.. code-block:: text

   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Performance improvement
   
   ## Testing
   - [ ] Unit tests added/updated
   - [ ] Integration tests pass
   - [ ] Manual testing completed
   
   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Documentation updated
   - [ ] Tests added for new functionality

Testing
-------

Running Tests
~~~~~~~~~~~~~

Run the full test suite:

.. code-block:: bash

   # Run all tests
   pytest tests/
   
   # Run specific test file
   pytest tests/test_models.py
   
   # Run with coverage
   pytest --cov=lib --cov=dataloader tests/

Writing Tests
~~~~~~~~~~~~~

All new code should include tests:

.. code-block:: python

   import pytest
   import torch
   from lib.models.new_model import NewModel
   
   class TestNewModel:
       def test_model_initialization(self):
           """Test model initializes correctly."""
           config = {"hidden_dim": 512}
           model = NewModel(config)
           assert model.hidden_dim == 512
           
       def test_forward_pass(self):
           """Test model forward pass."""
           model = NewModel({"hidden_dim": 512})
           inputs = torch.randn(1, 10, 512)
           outputs = model(inputs)
           assert outputs.shape[0] == 1

Test Categories
~~~~~~~~~~~~~~~

* **Unit Tests**: Test individual functions and classes
* **Integration Tests**: Test component interactions
* **End-to-End Tests**: Test complete workflows
* **Performance Tests**: Test speed and memory usage

Documentation
-------------

Documentation Structure
~~~~~~~~~~~~~~~~~~~~~~~

Documentation is organized as follows:

.. code-block:: text

   docs/
   ├── index.rst           # Main documentation page
   ├── installation.rst    # Installation guide
   ├── usage.rst          # Usage examples
   ├── api/               # API documentation
   │   ├── models.rst
   │   ├── dataloader.rst
   │   └── lib.rst
   └── _static/           # Static assets

Writing Documentation
~~~~~~~~~~~~~~~~~~~~~

* Use clear, concise language
* Include code examples for all features
* Add cross-references between related sections
* Update API documentation when changing code

.. code-block:: rst

   Example Function
   ~~~~~~~~~~~~~~~~
   
   .. autofunction:: lib.example.example_function
   
   Usage example:
   
   .. code-block:: python
   
      result = example_function(param1="value", param2=42)
      print(result)

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

Build documentation locally:

.. code-block:: bash

   cd docs
   make html
   
   # View in browser
   open _build/html/index.html

Code Review Guidelines
----------------------

Reviewing Code
~~~~~~~~~~~~~~

When reviewing pull requests:

* Check code correctness and style
* Verify tests are comprehensive
* Ensure documentation is updated
* Test functionality manually if needed
* Provide constructive feedback

Review Checklist:

* [ ] Code follows project style guidelines
* [ ] Functionality works as intended
* [ ] Tests cover new/changed code
* [ ] Documentation updated appropriately
* [ ] No performance regressions
* [ ] Security considerations addressed

Responding to Reviews
~~~~~~~~~~~~~~~~~~~~~

When receiving code review feedback:

* Address all comments promptly
* Ask for clarification if feedback is unclear
* Update code, tests, and documentation as needed
* Thank reviewers for their time and feedback

Community Guidelines
--------------------

Code of Conduct
~~~~~~~~~~~~~~~

* Be respectful and inclusive
* Welcome newcomers and help them learn
* Focus on constructive feedback
* Credit others for their contributions

Communication Channels
~~~~~~~~~~~~~~~~~~~~~~

* **GitHub Issues**: Bug reports and feature requests
* **Pull Requests**: Code contributions and discussions
* **Documentation**: Questions about usage and APIs

Getting Help
~~~~~~~~~~~~

If you need help:

1. Check existing documentation and issues
2. Create a detailed issue describing your problem
3. Include relevant code examples and error messages
4. Be patient and respectful when asking for help

Release Process
---------------

Versioning
~~~~~~~~~~

We follow semantic versioning (SemVer):

* ``MAJOR.MINOR.PATCH``
* Major: Breaking changes
* Minor: New features (backward compatible)
* Patch: Bug fixes (backward compatible)

Release Checklist
~~~~~~~~~~~~~~~~~

Before creating a release:

* [ ] All tests pass
* [ ] Documentation updated
* [ ] Version numbers updated
* [ ] Changelog updated
* [ ] Release notes prepared

Recognition
-----------

Contributors are recognized in:

* README.md contributors section
* Release notes
* Documentation acknowledgments
* Git commit history

Thank you for contributing to DLHM VidSGG!
