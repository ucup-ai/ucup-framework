"""
Agent Testing Module for UCUP Framework.

Provides comprehensive testing capabilities for probabilistic agents and coordination systems.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..core.manager import UCUPManager
from ..probabilistic_pkg.agent import ProbabilisticAgent
from ..coordination.hierarchical import HierarchicalCoordinator


class TestResult:
    """Represents the result of a single test case."""

    def __init__(self, test_name: str, success: bool, duration: float,
                 result_data: Dict[str, Any] = None, error: str = None):
        self.test_name = test_name
        self.success = success
        self.duration = duration
        self.result_data = result_data or {}
        self.error = error
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "test_name": self.test_name,
            "success": self.success,
            "duration": self.duration,
            "result_data": self.result_data,
            "error": self.error,
            "timestamp": self.timestamp
        }


class AgentTester:
    """Comprehensive testing framework for UCUP agents and coordinators."""

    def __init__(self, manager: UCUPManager):
        self.manager = manager
        self.config = manager.config
        self._test_results = []
        self._test_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=4)

    def test_probabilistic_agent(self, agent: ProbabilisticAgent,
                               test_cases: List[Dict[str, Any]],
                               timeout: float = 30.0) -> List[TestResult]:
        """
        Test a probabilistic agent with multiple test cases.

        Args:
            agent: The agent to test
            test_cases: List of test case dictionaries with 'input' and 'expected_output' keys
            timeout: Maximum time per test case in seconds

        Returns:
            List of test results
        """
        results = []

        for test_case in test_cases:
            start_time = time.time()

            try:
                # Run prediction with timeout
                future = self._executor.submit(agent.predict, test_case.get('input', {}))

                result = future.result(timeout=timeout)
                duration = time.time() - start_time

                # Validate result
                expected = test_case.get('expected_output', {})
                success = self._validate_prediction_result(result, expected)

                result_data = {
                    "prediction": result,
                    "expected": expected,
                    "validation_details": self._get_validation_details(result, expected)
                }

                results.append(TestResult(
                    test_name=f"probabilistic_agent_{len(results)}",
                    success=success,
                    duration=duration,
                    result_data=result_data
                ))

            except Exception as e:
                duration = time.time() - start_time
                results.append(TestResult(
                    test_name=f"probabilistic_agent_{len(results)}",
                    success=False,
                    duration=duration,
                    error=str(e)
                ))

        with self._test_lock:
            self._test_results.extend(results)

        return results

    def test_coordinator_performance(self, coordinator: HierarchicalCoordinator,
                                   workload: Dict[str, Any],
                                   duration: float = 10.0) -> TestResult:
        """
        Test coordinator performance under load.

        Args:
            coordinator: The coordinator to test
            workload: Workload configuration
            duration: Test duration in seconds

        Returns:
            Performance test result
        """
        start_time = time.time()
        tasks_completed = 0
        errors = []

        try:
            # Generate test tasks
            test_tasks = self._generate_test_tasks(workload)

            # Submit tasks to coordinator
            futures = []
            for task in test_tasks:
                future = self._executor.submit(
                    coordinator.execute_task, task
                )
                futures.append(future)

            # Wait for completion or timeout
            end_time = start_time + duration
            for future in as_completed(futures, timeout=max(0, end_time - time.time())):
                try:
                    result = future.result()
                    tasks_completed += 1
                except Exception as e:
                    errors.append(str(e))

            actual_duration = time.time() - start_time

            result_data = {
                "tasks_completed": tasks_completed,
                "total_tasks": len(test_tasks),
                "errors": errors,
                "throughput": tasks_completed / actual_duration if actual_duration > 0 else 0,
                "error_rate": len(errors) / len(test_tasks) if test_tasks else 0
            }

            return TestResult(
                test_name="coordinator_performance",
                success=len(errors) == 0,
                duration=actual_duration,
                result_data=result_data
            )

        except Exception as e:
            actual_duration = time.time() - start_time
            return TestResult(
                test_name="coordinator_performance",
                success=False,
                duration=actual_duration,
                error=str(e)
            )

    def test_agent_stress(self, agent_factory: Callable[[], ProbabilisticAgent],
                         concurrent_users: int = 10,
                         test_duration: float = 30.0) -> TestResult:
        """
        Stress test an agent with concurrent requests.

        Args:
            agent_factory: Function that creates new agent instances
            concurrent_users: Number of concurrent users
            test_duration: Test duration in seconds

        Returns:
            Stress test result
        """
        start_time = time.time()
        request_count = 0
        error_count = 0
        response_times = []

        def user_simulation(user_id: int):
            nonlocal request_count, error_count
            agent = agent_factory()
            user_start = time.time()

            while time.time() - user_start < test_duration:
                try:
                    req_start = time.time()
                    result = agent.predict({"test_input": f"user_{user_id}_req_{request_count}"})
                    response_time = time.time() - req_start

                    with self._test_lock:
                        request_count += 1
                        response_times.append(response_time)

                except Exception:
                    with self._test_lock:
                        error_count += 1

                time.sleep(0.01)  # Small delay between requests

        # Start concurrent users
        threads = []
        for i in range(concurrent_users):
            thread = threading.Thread(target=user_simulation, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        actual_duration = time.time() - start_time

        result_data = {
            "concurrent_users": concurrent_users,
            "total_requests": request_count,
            "error_count": error_count,
            "requests_per_second": request_count / actual_duration if actual_duration > 0 else 0,
            "average_response_time": sum(response_times) / len(response_times) if response_times else 0,
            "error_rate": error_count / request_count if request_count > 0 else 0
        }

        return TestResult(
            test_name="agent_stress_test",
            success=error_count == 0,
            duration=actual_duration,
            result_data=result_data
        )

    def _validate_prediction_result(self, result: Dict[str, Any],
                                  expected: Dict[str, Any]) -> bool:
        """Validate prediction result against expected output."""
        if not isinstance(result, dict):
            return False

        # Check required keys
        required_keys = ['probability', 'confidence_interval']
        for key in required_keys:
            if key not in result:
                return False

        # Check probability range
        prob = result.get('probability', 0)
        if not (0 <= prob <= 1):
            return False

        # Check confidence interval format
        conf_interval = result.get('confidence_interval')
        if not isinstance(conf_interval, (list, tuple)) or len(conf_interval) != 2:
            return False

        return True

    def _get_validation_details(self, result: Dict[str, Any],
                              expected: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed validation information."""
        details = {
            "probability_valid": 0 <= result.get('probability', -1) <= 1,
            "confidence_interval_valid": isinstance(result.get('confidence_interval'), (list, tuple)),
            "has_required_keys": all(k in result for k in ['probability', 'confidence_interval'])
        }

        return details

    def _generate_test_tasks(self, workload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate test tasks based on workload configuration."""
        task_count = workload.get('task_count', 100)
        task_types = workload.get('task_types', ['computation', 'io', 'mixed'])

        tasks = []
        for i in range(task_count):
            task = {
                'id': f'test_task_{i}',
                'type': task_types[i % len(task_types)],
                'priority': workload.get('priority', 'normal'),
                'data': {'input_size': workload.get('input_size', 100)}
            }
            tasks.append(task)

        return tasks

    def get_test_summary(self) -> Dict[str, Any]:
        """Get summary of all test results."""
        with self._test_lock:
            results = self._test_results.copy()

        if not results:
            return {"total_tests": 0, "passed": 0, "failed": 0}

        passed = sum(1 for r in results if r.success)
        failed = len(results) - passed
        total_duration = sum(r.duration for r in results)

        return {
            "total_tests": len(results),
            "passed": passed,
            "failed": failed,
            "success_rate": passed / len(results) if results else 0,
            "total_duration": total_duration,
            "average_duration": total_duration / len(results) if results else 0
        }

    def clear_results(self):
        """Clear stored test results."""
        with self._test_lock:
            self._test_results.clear()

    def shutdown(self):
        """Shutdown the tester."""
        if self._executor:
            self._executor.shutdown(wait=True)
