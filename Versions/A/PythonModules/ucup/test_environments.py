"""
Testing Environments for UCUP Framework

Provides comprehensive testing infrastructure with automated conda environment
management, multi-environment testing, and intelligent test generation for
reliable agent development and validation.

Copyright (c) 2025 UCUP Framework Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import asyncio
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class TestEnvironment:
    """Configuration for a test environment."""

    name: str
    python_version: str = "3.11"
    conda_packages: List[str] = field(
        default_factory=lambda: ["numpy", "scipy", "pandas", "pytest"]
    )
    pip_packages: List[str] = field(default_factory=lambda: ["ucup"])
    environment_variables: Dict[str, str] = field(default_factory=dict)
    requirements_file: Optional[str] = None
    test_directories: List[str] = field(default_factory=lambda: ["tests"])
    coverage_targets: List[str] = field(default_factory=lambda: ["src"])
    timeout_minutes: int = 30


@dataclass
class TestResult:
    """Result of a test execution."""

    environment_name: str
    test_suite: str
    success: bool
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    coverage_percentage: Optional[float] = None
    execution_time_seconds: float = 0.0
    output: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class TestSuite:
    """Configuration for a test suite."""

    name: str
    description: str
    test_files: List[str]
    environment_requirements: TestEnvironment
    pre_test_commands: List[str] = field(default_factory=list)
    post_test_commands: List[str] = field(default_factory=list)
    parallel_execution: bool = False
    max_parallel_workers: int = 4
    generate_coverage: bool = True
    generate_html_report: bool = True


class EnvironmentManager(ABC):
    """Abstract base class for environment managers."""

    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path.cwd()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    async def create_environment(self, env: TestEnvironment) -> bool:
        """Create a test environment."""
        pass

    @abstractmethod
    async def destroy_environment(self, env_name: str) -> bool:
        """Destroy a test environment."""
        pass

    @abstractmethod
    async def environment_exists(self, env_name: str) -> bool:
        """Check if environment exists."""
        pass

    @abstractmethod
    async def run_command(
        self, env_name: str, command: str, cwd: Optional[Path] = None
    ) -> subprocess.CompletedProcess:
        """Run command in environment."""
        pass

    @abstractmethod
    async def list_environments(self) -> List[str]:
        """List all environments."""
        pass


class CondaEnvironmentManager(EnvironmentManager):
    """Conda-based environment manager."""

    def __init__(
        self, base_path: Optional[Path] = None, conda_executable: str = "conda"
    ):
        super().__init__(base_path)
        self.conda_executable = conda_executable
        self.environments_dir = self.base_path / ".ucup" / "envs"

    async def create_environment(self, env: TestEnvironment) -> bool:
        """Create a conda environment."""
        try:
            self.environments_dir.mkdir(parents=True, exist_ok=True)

            # Create environment.yml
            env_file = self.environments_dir / f"{env.name}.yml"
            env_config = {
                "name": env.name,
                "channels": ["conda-forge", "defaults"],
                "dependencies": [
                    f"python={env.python_version}",
                    "pip",
                    *env.conda_packages,
                ],
            }

            if env.pip_packages:
                env_config["dependencies"].append({"pip": env.pip_packages})

            import yaml

            with open(env_file, "w") as f:
                yaml.dump(env_config, f, default_flow_style=False)

            # Create conda environment
            cmd = [
                self.conda_executable,
                "env",
                "create",
                "--file",
                str(env_file),
                "--prefix",
                str(self.environments_dir / env.name),
            ]

            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.base_path,
            )

            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                self.logger.info(f"Created conda environment: {env.name}")
                return True
            else:
                self.logger.error(
                    f"Failed to create environment {env.name}: {stderr.decode()}"
                )
                return False

        except Exception as e:
            self.logger.error(f"Error creating environment {env.name}: {e}")
            return False

    async def destroy_environment(self, env_name: str) -> bool:
        """Destroy a conda environment."""
        try:
            cmd = [self.conda_executable, "env", "remove", "--name", env_name, "--yes"]

            result = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await result.communicate()

            # Also remove local env directory if it exists
            env_path = self.environments_dir / env_name
            if env_path.exists():
                shutil.rmtree(env_path)

            if result.returncode == 0:
                self.logger.info(f"Destroyed conda environment: {env_name}")
                return True
            else:
                self.logger.error(
                    f"Failed to destroy environment {env_name}: {stderr.decode()}"
                )
                return False

        except Exception as e:
            self.logger.error(f"Error destroying environment {env_name}: {e}")
            return False

    async def environment_exists(self, env_name: str) -> bool:
        """Check if conda environment exists."""
        try:
            cmd = [self.conda_executable, "env", "list", "--json"]

            result = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                import json

                envs_data = json.loads(stdout.decode())
                env_names = [env["name"] for env in envs_data.get("envs", [])]
                return env_name in env_names
            else:
                # Check local env directory
                env_path = self.environments_dir / env_name
                return env_path.exists()

        except Exception:
            return False

    async def run_command(
        self, env_name: str, command: str, cwd: Optional[Path] = None
    ) -> subprocess.CompletedProcess:
        """Run command in conda environment."""
        # Activate conda environment and run command
        activate_cmd = f"conda activate {env_name} && {command}"

        # Use conda run for cross-platform compatibility
        cmd = [self.conda_executable, "run", "--name", env_name, "bash", "-c", command]

        result = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd or self.base_path,
        )

        stdout, stderr = await result.communicate()

        return subprocess.CompletedProcess(cmd, result.returncode, stdout, stderr)

    async def list_environments(self) -> List[str]:
        """List conda environments."""
        try:
            cmd = [self.conda_executable, "env", "list", "--json"]

            result = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                import json

                envs_data = json.loads(stdout.decode())
                return [env["name"] for env in envs_data.get("envs", [])]
            else:
                return []

        except Exception:
            return []


class VenvEnvironmentManager(EnvironmentManager):
    """Python venv-based environment manager."""

    def __init__(self, base_path: Optional[Path] = None):
        super().__init__(base_path)
        self.environments_dir = self.base_path / ".ucup" / "venvs"

    async def create_environment(self, env: TestEnvironment) -> bool:
        """Create a Python venv environment."""
        try:
            self.environments_dir.mkdir(parents=True, exist_ok=True)
            env_path = self.environments_dir / env.name

            # Create virtual environment
            cmd = [sys.executable, "-m", "venv", str(env_path)]

            result = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await result.communicate()

            if result.returncode != 0:
                self.logger.error(
                    f"Failed to create venv {env.name}: {stderr.decode()}"
                )
                return False

            # Install packages
            pip_cmd = [
                str(env_path / "bin" / "pip"),
                "install",
                "--upgrade",
                "pip",
                *env.pip_packages,
            ]

            if env.requirements_file:
                pip_cmd.extend(["-r", env.requirements_file])

            result = await asyncio.create_subprocess_exec(
                *pip_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                self.logger.info(f"Created venv environment: {env.name}")
                return True
            else:
                self.logger.error(
                    f"Failed to install packages in {env.name}: {stderr.decode()}"
                )
                return False

        except Exception as e:
            self.logger.error(f"Error creating venv {env.name}: {e}")
            return False

    async def destroy_environment(self, env_name: str) -> bool:
        """Destroy a venv environment."""
        try:
            env_path = self.environments_dir / env_name
            if env_path.exists():
                shutil.rmtree(env_path)
                self.logger.info(f"Destroyed venv environment: {env_name}")
                return True
            else:
                self.logger.warning(f"Environment {env_name} does not exist")
                return False

        except Exception as e:
            self.logger.error(f"Error destroying venv {env_name}: {e}")
            return False

    async def environment_exists(self, env_name: str) -> bool:
        """Check if venv environment exists."""
        env_path = self.environments_dir / env_name
        return env_path.exists() and (env_path / "bin" / "python").exists()

    async def run_command(
        self, env_name: str, command: str, cwd: Optional[Path] = None
    ) -> subprocess.CompletedProcess:
        """Run command in venv environment."""
        env_path = self.environments_dir / env_name
        python_executable = env_path / "bin" / "python"

        # Prepend python executable to command
        full_command = f'{python_executable} -c "{command}"'

        result = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            full_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd or self.base_path,
        )

        stdout, stderr = await result.communicate()

        return subprocess.CompletedProcess(
            ["bash", "-c", full_command], result.returncode, stdout, stderr
        )

    async def list_environments(self) -> List[str]:
        """List venv environments."""
        if not self.environments_dir.exists():
            return []

        envs = []
        for item in self.environments_dir.iterdir():
            if item.is_dir() and (item / "bin" / "python").exists():
                envs.append(item.name)

        return envs


class TestRunner:
    """Manages test execution across environments."""

    def __init__(self, env_manager: EnvironmentManager):
        self.env_manager = env_manager
        self.logger = logging.getLogger(__name__)

    async def run_test_suite(self, suite: TestSuite, env_name: str) -> TestResult:
        """Run a test suite in the specified environment."""
        start_time = datetime.now()

        result = TestResult(environment_name=env_name, test_suite=suite.name)

        try:
            # Check if environment exists
            if not await self.env_manager.environment_exists(env_name):
                result.errors.append(f"Environment {env_name} does not exist")
                result.success = False
                return result

            # Run pre-test commands
            for cmd in suite.pre_test_commands:
                self.logger.info(f"Running pre-test command: {cmd}")
                proc_result = await self.env_manager.run_command(env_name, cmd)
                if proc_result.returncode != 0:
                    result.warnings.append(f"Pre-test command failed: {cmd}")

            # Run tests
            test_commands = self._generate_test_commands(suite)

            all_output = []
            all_errors = []

            for cmd in test_commands:
                self.logger.info(f"Running test command: {cmd}")
                proc_result = await self.env_manager.run_command(env_name, cmd)

                if proc_result.stdout:
                    all_output.append(proc_result.stdout.decode())
                if proc_result.stderr:
                    all_errors.append(proc_result.stderr.decode())

                # Parse pytest output
                if "pytest" in cmd:
                    self._parse_pytest_output(proc_result.stdout.decode(), result)

            result.output = "\n".join(all_output)
            if all_errors:
                result.errors.extend(all_errors)

            # Run post-test commands
            for cmd in suite.post_test_commands:
                self.logger.info(f"Running post-test command: {cmd}")
                proc_result = await self.env_manager.run_command(env_name, cmd)
                if proc_result.returncode != 0:
                    result.warnings.append(f"Post-test command failed: {cmd}")

            # Determine success
            result.success = result.failed_tests == 0 and len(result.errors) == 0

        except Exception as e:
            result.errors.append(f"Test execution failed: {str(e)}")
            result.success = False

        finally:
            result.execution_time_seconds = (
                datetime.now() - start_time
            ).total_seconds()

        return result

    def _generate_test_commands(self, suite: TestSuite) -> List[str]:
        """Generate test commands for the suite."""
        commands = []

        for test_file in suite.test_files:
            if suite.parallel_execution:
                cmd = f"pytest {test_file} -n {suite.max_parallel_workers}"
            else:
                cmd = f"pytest {test_file}"

            if suite.generate_coverage:
                coverage_targets = " ".join(
                    f"--cov={target}" for target in suite.coverage_targets
                )
                cmd += f" {coverage_targets} --cov-report=term-missing"

            if suite.generate_html_report:
                cmd += " --html=test_report.html --self-contained-html"

            commands.append(cmd)

        return commands

    def _parse_pytest_output(self, output: str, result: TestResult):
        """Parse pytest output to extract test results."""
        import re

        # Parse summary line like "7 passed, 2 failed, 1 skipped"
        summary_match = re.search(
            r"(\d+)\s+passed(?:,\s*(\d+)\s+failed)?(?:,\s*(\d+)\s+skipped)?", output
        )
        if summary_match:
            result.passed_tests = int(summary_match.group(1))
            result.failed_tests = (
                int(summary_match.group(2)) if summary_match.group(2) else 0
            )
            result.skipped_tests = (
                int(summary_match.group(3)) if summary_match.group(3) else 0
            )
            result.total_tests = (
                result.passed_tests + result.failed_tests + result.skipped_tests
            )

        # Parse coverage percentage
        coverage_match = re.search(r"TOTAL\s+\d+\s+\d+\s+\d+\s+(\d+)%", output)
        if coverage_match:
            result.coverage_percentage = float(coverage_match.group(1))


class TestEnvironmentManager:
    """High-level manager for test environments and execution."""

    def __init__(self, base_path: Optional[Path] = None, prefer_conda: bool = True):
        self.base_path = base_path or Path.cwd()
        self.prefer_conda = prefer_conda

        # Initialize environment managers
        self.conda_manager = CondaEnvironmentManager(self.base_path)
        self.venv_manager = VenvEnvironmentManager(self.base_path)
        self.test_runner = TestRunner(
            self.conda_manager if prefer_conda else self.venv_manager
        )

        self.logger = logging.getLogger(__name__)

    async def setup_test_environments(
        self, environments: List[TestEnvironment]
    ) -> Dict[str, bool]:
        """Setup multiple test environments."""
        results = {}

        for env in environments:
            self.logger.info(f"Setting up environment: {env.name}")

            # Try conda first if preferred
            if self.prefer_conda:
                success = await self.conda_manager.create_environment(env)
                if success:
                    results[env.name] = True
                    continue

            # Fallback to venv
            success = await self.venv_manager.create_environment(env)
            results[env.name] = success

            if success:
                self.logger.info(f"Successfully created environment: {env.name}")
            else:
                self.logger.error(f"Failed to create environment: {env.name}")

        return results

    async def run_tests_across_environments(
        self, suites: List[TestSuite]
    ) -> List[TestResult]:
        """Run test suites across all available environments."""
        results = []

        # Get available environments
        conda_envs = await self.conda_manager.list_environments()
        venv_envs = await self.venv_manager.list_environments()
        all_envs = set(conda_envs + venv_envs)

        for suite in suites:
            for env_name in all_envs:
                self.logger.info(f"Running {suite.name} in {env_name}")

                # Switch test runner based on environment type
                if env_name in conda_envs:
                    self.test_runner = TestRunner(self.conda_manager)
                else:
                    self.test_runner = TestRunner(self.venv_manager)

                result = await self.test_runner.run_test_suite(suite, env_name)
                results.append(result)

                if result.success:
                    self.logger.info(f"✅ Tests passed in {env_name}")
                else:
                    self.logger.error(
                        f"❌ Tests failed in {env_name}: {len(result.errors)} errors"
                    )

        return results

    async def generate_test_scaffold(self, project_type: str, domain: str) -> TestSuite:
        """Generate test scaffolding for a project."""
        suite_name = f"{project_type}_{domain}_tests"

        # Generate test files based on project type
        test_files = self._generate_test_files(project_type, domain)

        # Create environment requirements
        env = TestEnvironment(
            name=f"{project_type}_{domain}_env",
            python_version="3.11",
            conda_packages=[
                "numpy",
                "scipy",
                "pandas",
                "pytest",
                "pytest-cov",
                "pytest-asyncio",
            ],
            pip_packages=["ucup", "fastapi", "uvicorn"]
            if project_type == "api"
            else ["ucup"],
        )

        suite = TestSuite(
            name=suite_name,
            description=f"Comprehensive tests for {project_type} {domain} project",
            test_files=test_files,
            environment_requirements=env,
            pre_test_commands=["pip install -e .", "mkdir -p test_results"],
            post_test_commands=[
                "rm -rf .pytest_cache",
                "mv test_report.html test_results/ 2>/dev/null || true",
            ],
            parallel_execution=True,
            generate_coverage=True,
            generate_html_report=True,
        )

        return suite

    def _generate_test_files(self, project_type: str, domain: str) -> List[str]:
        """Generate test file paths based on project type and domain."""
        base_tests = ["tests/test_basic.py", "tests/test_integration.py"]

        if project_type == "agent":
            base_tests.extend(
                [
                    "tests/test_uncertainty.py",
                    "tests/test_probabilistic_reasoning.py",
                    f"tests/test_{domain}_specific.py",
                ]
            )

        elif project_type == "api":
            base_tests.extend(
                [
                    "tests/test_endpoints.py",
                    "tests/test_middleware.py",
                    "tests/test_authentication.py",
                ]
            )

        elif project_type == "multimodal":
            base_tests.extend(
                ["tests/test_vision.py", "tests/test_audio.py", "tests/test_fusion.py"]
            )

        return base_tests

    async def cleanup_environments(
        self, env_names: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """Clean up test environments."""
        results = {}

        if env_names is None:
            # Clean up all environments
            conda_envs = await self.conda_manager.list_environments()
            venv_envs = await self.venv_manager.list_environments()
            env_names = conda_envs + venv_envs

        for env_name in env_names:
            # Try conda first
            success = await self.conda_manager.destroy_environment(env_name)
            if not success:
                # Try venv
                success = await self.venv_manager.destroy_environment(env_name)

            results[env_name] = success

            if success:
                self.logger.info(f"Cleaned up environment: {env_name}")
            else:
                self.logger.warning(f"Failed to clean up environment: {env_name}")

        return results


def create_default_test_environments() -> List[TestEnvironment]:
    """Create default test environments for UCUP."""
    return [
        TestEnvironment(
            name="ucup-py311",
            python_version="3.11",
            conda_packages=[
                "numpy",
                "scipy",
                "pandas",
                "matplotlib",
                "pytest",
                "pytest-cov",
                "pytest-asyncio",
                "pytest-benchmark",
                "black",
                "isort",
                "flake8",
                "mypy",
            ],
            pip_packages=["ucup", "fastapi", "uvicorn", "plotly"],
            environment_variables={"UCUP_TEST_MODE": "true", "PYTHONPATH": "/app/src"},
        ),
        TestEnvironment(
            name="ucup-py310",
            python_version="3.10",
            conda_packages=[
                "numpy",
                "scipy",
                "pandas",
                "matplotlib",
                "pytest",
                "pytest-cov",
                "pytest-asyncio",
                "pytest-benchmark",
            ],
            pip_packages=["ucup", "fastapi", "uvicorn"],
            environment_variables={"UCUP_TEST_MODE": "true", "PYTHONPATH": "/app/src"},
        ),
        TestEnvironment(
            name="ucup-minimal",
            python_version="3.9",
            conda_packages=["numpy", "pytest", "pytest-cov"],
            pip_packages=["ucup"],
            environment_variables={"UCUP_TEST_MODE": "true"},
        ),
    ]


async def setup_ucup_test_environment(env_name: str = "ucup-test") -> bool:
    """Setup a UCUP test environment with all necessary dependencies."""
    manager = TestEnvironmentManager()

    test_env = TestEnvironment(
        name=env_name,
        python_version="3.11",
        conda_packages=[
            "numpy",
            "scipy",
            "pandas",
            "matplotlib",
            "plotly",
            "pytest",
            "pytest-cov",
            "pytest-asyncio",
            "pytest-benchmark",
            "pytest-xdist",
            "black",
            "isort",
            "flake8",
            "mypy",
            "jupyter",
            "notebook",
            "ipykernel",
        ],
        pip_packages=[
            "ucup",
            "fastapi",
            "uvicorn",
            "pydantic",
            "requests",
            "aiohttp",
            "httpx",
            "python-multipart",
            "python-jose[cryptography]",
            "uvloop",
            "gunicorn",
        ],
        environment_variables={
            "UCUP_TEST_MODE": "true",
            "PYTHONPATH": "/app/src",
            "UCUP_LOG_LEVEL": "INFO",
        },
    )

    return await manager.setup_test_environments([test_env])


async def run_ucup_tests(
    test_path: str = "tests", env_name: str = "ucup-test"
) -> TestResult:
    """Run UCUP tests in the specified environment."""
    manager = TestEnvironmentManager()

    # Create test suite
    test_files = (
        [str(f) for f in Path(test_path).glob("test_*.py")]
        if Path(test_path).exists()
        else []
    )

    if not test_files:
        test_files = ["tests/test_basic.py"]  # Fallback

    suite = TestSuite(
        name="ucup_test_suite",
        description="UCUP comprehensive test suite",
        test_files=test_files,
        environment_requirements=TestEnvironment(
            name=env_name,
            python_version="3.11",
            conda_packages=["pytest", "pytest-cov", "pytest-asyncio"],
            pip_packages=["ucup"],
        ),
        parallel_execution=True,
        generate_coverage=True,
        generate_html_report=True,
    )

    runner = TestRunner(manager.conda_manager)
    return await runner.run_test_suite(suite, env_name)


def generate_test_template(
    agent_name: str, domain: str, output_dir: str = "tests"
) -> str:
    """Generate a test template for a UCUP agent."""
    template = f'''"""
Tests for {agent_name} - {domain} domain agent.

Generated by UCUP Test Environment Manager.
"""

import pytest
import asyncio
from typing import Dict, Any
from pathlib import Path

# Import the agent (adjust import path as needed)
try:
    from src.{agent_name.lower().replace(" ", "_")} import {agent_name.replace(" ", "")}
except ImportError:
    # Mock agent for testing
    class {agent_name.replace(" ", "")}:
        async def execute(self, task: str, **kwargs):
            return {{
                "value": f"Mock response for: {{task}}",
                "confidence": 0.85,
                "uncertainty_score": 0.15,
                "robustness_score": 0.82
            }}

class Test{agent_name.replace(" ", "")}:
    """Test suite for {agent_name}."""

    @pytest.fixture
    async def agent(self):
        """Create agent instance for testing."""
        return {agent_name.replace(" ", "")}()

    @pytest.mark.asyncio
    async def test_agent_creation(self, agent):
        """Test agent can be created successfully."""
        assert agent is not None
        assert hasattr(agent, 'execute')

    @pytest.mark.asyncio
    async def test_basic_execution(self, agent):
        """Test basic agent execution."""
        task = "Test task for {domain} domain"
        result = await agent.execute(task)

        assert isinstance(result, dict)
        assert "value" in result
        assert "confidence" in result
        assert isinstance(result["confidence"], (int, float))
        assert 0.0 <= result["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_uncertainty_quantification(self, agent):
        """Test uncertainty quantification capabilities."""
        task = "Complex {domain} task requiring uncertainty analysis"
        result = await agent.execute(task)

        assert "uncertainty_score" in result
        assert "robustness_score" in result
        assert isinstance(result["uncertainty_score"], (int, float))
        assert isinstance(result["robustness_score"], (int, float))
        assert result["uncertainty_score"] >= 0.0
        assert result["robustness_score"] >= 0.0

    @pytest.mark.asyncio
    async def test_domain_specific_behavior(self, agent):
        """Test domain-specific behavior and adaptations."""
        domain_tasks = [
            "{domain} analysis task",
            "{domain} optimization problem",
            "{domain} decision making scenario"
        ]

        for task in domain_tasks:
            result = await agent.execute(task)
            assert result["value"] is not None
            assert len(str(result["value"])) > 0

    @pytest.mark.asyncio
    async def test_error_handling(self, agent):
        """Test error handling capabilities."""
        # Test with invalid input
        try:
            result = await agent.execute("")
            # Should handle empty input gracefully
            assert result is not None
        except Exception as e:
            # If it raises an exception, it should be a controlled one
            assert "invalid" in str(e).lower() or "empty" in str(e).lower()

    @pytest.mark.asyncio
    async def test_performance_constraints(self, agent):
        """Test performance meets basic constraints."""
        import time

        start_time = time.time()
        result = await agent.execute("Performance test task")
        execution_time = time.time() - start_time

        # Should complete within reasonable time (30 seconds)
        assert execution_time < 30.0
        assert result is not None

    @pytest.mark.parametrize("task_type", [
        "simple_query",
        "complex_analysis",
        "decision_making",
        "optimization"
    ])
    @pytest.mark.asyncio
    async def test_task_types(self, agent, task_type):
        """Test different types of tasks."""
        task_descriptions = {{
            "simple_query": "What is the current status?",
            "complex_analysis": "Analyze the following data and provide insights",
            "decision_making": "Choose the best option from these alternatives",
            "optimization": "Optimize this process for maximum efficiency"
        }}

        task = task_descriptions[task_type]
        result = await agent.execute(task)

        assert result["value"] is not None
        assert result["confidence"] > 0.1  # Should have some confidence

@pytest.mark.benchmark
class Test{agent_name.replace(" ", "")}Performance:
    """Performance benchmarks for {agent_name}."""

    @pytest.fixture
    async def agent(self):
        """Create agent instance for benchmarking."""
        return {agent_name.replace(" ", "")}()

    @pytest.mark.asyncio
    async def test_execution_speed(self, agent, benchmark):
        """Benchmark agent execution speed."""
        async def run_task():
            return await agent.execute("Benchmark task")

        result = benchmark.pedantic(run_task, iterations=10, rounds=5)
        assert result is not None

    @pytest.mark.asyncio
    async def test_memory_usage(self, agent, benchmark):
        """Benchmark memory usage."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        async def run_task():
            return await agent.execute("Memory benchmark task")

        result = benchmark.pedantic(run_task, iterations=5, rounds=3)

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (< 100MB)
        assert memory_increase < 100 * 1024 * 1024
        assert result is not None

if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
'''

    # Create output directory and write template
    Path(output_dir).mkdir(exist_ok=True)
    test_file = Path(output_dir) / f'test_{agent_name.lower().replace(" ", "_")}.py'

    with open(test_file, "w") as f:
        f.write(template)

    return str(test_file)
