# Testing GitHub Actions Locally with `act`

## Prerequisites
- Docker Desktop installed and running
- `act` CLI tool installed

## Installation
```powershell
# Windows (Chocolatey)
choco install act-cli

# Or download from: https://github.com/nektos/act/releases
```

## Basic Testing Commands

### 1. Dry run (check syntax without execution)
```bash
act --dryrun
```

### 2. Test specific workflow file
```bash
act -W .github/workflows/docs.yml
```

### 3. Test specific job
```bash
act -j build
```

### 4. Test with secrets (if needed)
```bash
act -s GITHUB_TOKEN=your_token_here
```

### 5. Test pull request event
```bash
act pull_request
```

## For Your Documentation Workflow

Test your docs workflow specifically:
```bash
# Test the build job only
act -j build -W .github/workflows/docs.yml

# Test with verbose output
act -j build -W .github/workflows/docs.yml -v
```

## Limitations
- Pages deployment won't work locally (GitHub-specific)
- Some features may behave differently
- Use smaller Docker images for faster testing

## Recommended Workflow
1. Make changes to `.github/workflows/docs.yml`
2. Run `act --dryrun` to check syntax
3. Run `act -j build` to test build process
4. Only push when local tests pass
