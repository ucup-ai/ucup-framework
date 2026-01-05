"""
Enhanced deployment and orchestration capabilities for UCUP Framework.

This module provides production-ready deployment tools including container orchestration,
health monitoring, auto-scaling, and operational management for agent networks.
"""

import asyncio
import json
import logging
import os
import signal
import subprocess
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import psutil


class DeploymentState(Enum):
    """Deployment states."""

    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class ScalingStrategy(Enum):
    """Auto-scaling strategies."""

    CPU_BASED = "cpu_based"
    MEMORY_BASED = "memory_based"
    REQUEST_BASED = "request_based"
    CUSTOM_METRIC = "custom_metric"


@dataclass
class HealthCheck:
    """Health check configuration."""

    name: str
    endpoint: str
    method: str = "GET"
    interval: int = 30  # seconds
    timeout: int = 10  # seconds
    failure_threshold: int = 3
    success_threshold: int = 1
    headers: Dict[str, str] = field(default_factory=dict)
    expected_status: int = 200


@dataclass
class ScalingConfig:
    """Auto-scaling configuration."""

    enabled: bool = True
    min_instances: int = 1
    max_instances: int = 10
    scale_up_threshold: float = 0.8  # 80% utilization
    scale_down_threshold: float = 0.3  # 30% utilization
    cooldown_period: int = 300  # seconds
    strategy: ScalingStrategy = ScalingStrategy.CPU_BASED
    custom_metric: Optional[str] = None


@dataclass
class ContainerConfig:
    """Container deployment configuration."""

    image: str
    ports: List[str] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)
    volumes: List[str] = field(default_factory=list)
    networks: List[str] = field(default_factory=list)
    restart_policy: str = "unless-stopped"
    healthcheck: Optional[HealthCheck] = None


@dataclass
class DeploymentSpec:
    """Complete deployment specification."""

    name: str
    version: str = "latest"
    environment: str = "development"
    replicas: int = 1
    container: ContainerConfig = None
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    health_checks: List[HealthCheck] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)


class DeploymentProvider(ABC):
    """Abstract base class for deployment providers."""

    @abstractmethod
    async def deploy(self, spec: DeploymentSpec) -> str:
        """Deploy the application."""
        pass

    @abstractmethod
    async def undeploy(self, deployment_id: str) -> bool:
        """Remove the deployment."""
        pass

    @abstractmethod
    async def get_status(self, deployment_id: str) -> DeploymentState:
        """Get deployment status."""
        pass

    @abstractmethod
    async def scale(self, deployment_id: str, replicas: int) -> bool:
        """Scale the deployment."""
        pass

    @abstractmethod
    async def get_logs(self, deployment_id: str, lines: int = 100) -> str:
        """Get deployment logs."""
        pass


class DockerDeploymentProvider(DeploymentProvider):
    """Docker-based deployment provider."""

    def __init__(self, docker_host: str = "unix:///var/run/docker.sock"):
        self.docker_host = docker_host
        self.containers: Dict[str, str] = {}  # deployment_id -> container_id

    async def deploy(self, spec: DeploymentSpec) -> str:
        """Deploy using Docker."""
        deployment_id = f"{spec.name}-{spec.version}-{int(time.time())}"

        # Build Docker command
        cmd = ["docker", "run", "-d", "--name", deployment_id]

        # Add ports
        for port in spec.container.ports:
            cmd.extend(["-p", port])

        # Add environment variables
        for key, value in spec.container.environment.items():
            cmd.extend(["-e", f"{key}={value}"])

        # Add volumes
        for volume in spec.container.volumes:
            cmd.extend(["-v", volume])

        # Add networks
        for network in spec.container.networks:
            cmd.extend(["--network", network])

        # Add restart policy
        cmd.extend(["--restart", spec.container.restart_policy])

        # Add image
        cmd.append(spec.container.image)

        # Execute command
        result = await self._run_command(cmd)
        if result.returncode == 0:
            container_id = result.stdout.strip()
            self.containers[deployment_id] = container_id
            return deployment_id
        else:
            raise RuntimeError(f"Docker deployment failed: {result.stderr}")

    async def undeploy(self, deployment_id: str) -> bool:
        """Remove Docker container."""
        if deployment_id not in self.containers:
            return False

        container_id = self.containers[deployment_id]
        cmd = ["docker", "rm", "-f", container_id]

        result = await self._run_command(cmd)
        if result.returncode == 0:
            del self.containers[deployment_id]
            return True
        return False

    async def get_status(self, deployment_id: str) -> DeploymentState:
        """Get container status."""
        if deployment_id not in self.containers:
            return DeploymentState.STOPPED

        container_id = self.containers[deployment_id]
        cmd = ["docker", "inspect", "-f", "{{.State.Status}}", container_id]

        result = await self._run_command(cmd)
        if result.returncode == 0:
            status = result.stdout.strip()
            status_map = {
                "created": DeploymentState.CREATED,
                "running": DeploymentState.RUNNING,
                "exited": DeploymentState.STOPPED,
                "dead": DeploymentState.FAILED,
            }
            return status_map.get(status, DeploymentState.FAILED)
        return DeploymentState.FAILED

    async def scale(self, deployment_id: str, replicas: int) -> bool:
        """Scale Docker deployment (simplified - creates multiple containers)."""
        # This is a simplified implementation
        # In practice, you'd use Docker Compose or Swarm for scaling
        current_status = await self.get_status(deployment_id)
        if current_status != DeploymentState.RUNNING:
            return False

        # For now, just return success
        return True

    async def get_logs(self, deployment_id: str, lines: int = 100) -> str:
        """Get container logs."""
        if deployment_id not in self.containers:
            return ""

        container_id = self.containers[deployment_id]
        cmd = ["docker", "logs", "--tail", str(lines), container_id]

        result = await self._run_command(cmd)
        return result.stdout if result.returncode == 0 else result.stderr

    async def _run_command(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """Run a command asynchronously."""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            future = loop.run_in_executor(
                executor, subprocess.run, cmd, {"capture_output": True, "text": True}
            )
            return await future


class KubernetesDeploymentProvider(DeploymentProvider):
    """Kubernetes-based deployment provider."""

    def __init__(self, namespace: str = "default", kubeconfig: Optional[str] = None):
        self.namespace = namespace
        self.kubeconfig = kubeconfig

    async def deploy(self, spec: DeploymentSpec) -> str:
        """Deploy to Kubernetes."""
        deployment_id = f"{spec.name}-{spec.version}"

        # Generate Kubernetes manifests
        manifests = self._generate_k8s_manifests(spec)

        # Apply manifests using kubectl
        for manifest in manifests:
            cmd = ["kubectl", "apply", "-f", "-"]
            if self.kubeconfig:
                cmd.extend(["--kubeconfig", self.kubeconfig])

            result = await self._run_command(cmd, input=manifest)
            if result.returncode != 0:
                raise RuntimeError(f"Kubernetes deployment failed: {result.stderr}")

        return deployment_id

    async def undeploy(self, deployment_id: str) -> bool:
        """Remove from Kubernetes."""
        cmd = ["kubectl", "delete", "deployment", deployment_id, "-n", self.namespace]
        if self.kubeconfig:
            cmd.extend(["--kubeconfig", self.kubeconfig])

        result = await self._run_command(cmd)
        return result.returncode == 0

    async def get_status(self, deployment_id: str) -> DeploymentState:
        """Get deployment status from Kubernetes."""
        cmd = [
            "kubectl",
            "get",
            "deployment",
            deployment_id,
            "-n",
            self.namespace,
            "-o",
            "jsonpath={.status.conditions[0].type}",
        ]
        if self.kubeconfig:
            cmd.extend(["--kubeconfig", self.kubeconfig])

        result = await self._run_command(cmd)
        if result.returncode == 0:
            status = result.stdout.strip()
            if status == "Available":
                return DeploymentState.RUNNING
            elif status == "Progressing":
                return DeploymentState.STARTING
            else:
                return DeploymentState.FAILED
        return DeploymentState.FAILED

    async def scale(self, deployment_id: str, replicas: int) -> bool:
        """Scale Kubernetes deployment."""
        cmd = [
            "kubectl",
            "scale",
            "deployment",
            deployment_id,
            f"--replicas={replicas}",
            "-n",
            self.namespace,
        ]
        if self.kubeconfig:
            cmd.extend(["--kubeconfig", self.kubeconfig])

        result = await self._run_command(cmd)
        return result.returncode == 0

    async def get_logs(self, deployment_id: str, lines: int = 100) -> str:
        """Get logs from Kubernetes pods."""
        cmd = [
            "kubectl",
            "logs",
            f"deployment/{deployment_id}",
            "-n",
            self.namespace,
            "--tail",
            str(lines),
        ]
        if self.kubeconfig:
            cmd.extend(["--kubeconfig", self.kubeconfig])

        result = await self._run_command(cmd)
        return result.stdout if result.returncode == 0 else result.stderr

    def _generate_k8s_manifests(self, spec: DeploymentSpec) -> List[str]:
        """Generate Kubernetes manifests."""
        manifests = []

        # Deployment manifest
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"{spec.name}-{spec.version}",
                "namespace": self.namespace,
                "labels": spec.labels,
            },
            "spec": {
                "replicas": spec.replicas,
                "selector": {"matchLabels": {"app": spec.name}},
                "template": {
                    "metadata": {"labels": {"app": spec.name, **spec.labels}},
                    "spec": {
                        "containers": [
                            {
                                "name": spec.name,
                                "image": spec.container.image,
                                "ports": [
                                    {"containerPort": int(port.split(":")[1])}
                                    for port in spec.container.ports
                                ],
                                "env": [
                                    {"name": k, "value": v}
                                    for k, v in spec.container.environment.items()
                                ],
                                "volumeMounts": [
                                    {"name": "config", "mountPath": "/app/config"}
                                ],
                            }
                        ],
                        "volumes": [
                            {
                                "name": "config",
                                "configMap": {"name": f"{spec.name}-config"},
                            }
                        ],
                    },
                },
            },
        }

        manifests.append(json.dumps(deployment, indent=2))

        # Service manifest if ports are exposed
        if spec.container.ports:
            service = {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {
                    "name": f"{spec.name}-service",
                    "namespace": self.namespace,
                },
                "spec": {
                    "selector": {"app": spec.name},
                    "ports": [
                        {
                            "port": int(port.split(":")[0]),
                            "targetPort": int(port.split(":")[1]),
                        }
                        for port in spec.container.ports
                    ],
                },
            }
            manifests.append(json.dumps(service, indent=2))

        return manifests

    async def _run_command(
        self, cmd: List[str], input: str = None
    ) -> subprocess.CompletedProcess:
        """Run a command asynchronously."""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            kwargs = {"capture_output": True, "text": True}
            if input:
                kwargs["input"] = input

            future = loop.run_in_executor(executor, subprocess.run, cmd, kwargs)
            return await future


class HealthMonitor:
    """Health monitoring for deployments."""

    def __init__(self):
        self.checks: Dict[str, HealthCheck] = {}
        self.results: Dict[str, List[Dict[str, Any]]] = {}
        self.logger = logging.getLogger(__name__)

    def add_health_check(self, check: HealthCheck):
        """Add a health check."""
        self.checks[check.name] = check
        self.results[check.name] = []

    async def run_health_checks(self) -> Dict[str, bool]:
        """Run all health checks."""
        results = {}

        for check_name, check in self.checks.items():
            is_healthy = await self._run_single_check(check)
            results[check_name] = is_healthy

            # Store result
            self.results[check_name].append(
                {
                    "timestamp": datetime.now(),
                    "healthy": is_healthy,
                    "response_time": None,  # Would measure actual response time
                }
            )

            # Keep only recent results
            if len(self.results[check_name]) > 10:
                self.results[check_name] = self.results[check_name][-10:]

        return results

    async def _run_single_check(self, check: HealthCheck) -> bool:
        """Run a single health check."""
        try:
            import aiohttp

            timeout = aiohttp.ClientTimeout(total=check.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.request(
                    check.method, check.endpoint, headers=check.headers
                ) as response:
                    return response.status == check.expected_status

        except Exception as e:
            self.logger.warning(f"Health check failed for {check.endpoint}: {e}")
            return False

    def get_health_history(self, check_name: str) -> List[Dict[str, Any]]:
        """Get health check history."""
        return self.results.get(check_name, [])

    def is_system_healthy(self, required_checks: List[str] = None) -> bool:
        """Check if the overall system is healthy."""
        checks_to_verify = required_checks or list(self.checks.keys())

        for check_name in checks_to_verify:
            if check_name not in self.results:
                return False

            # Check recent results for failures
            recent_results = self.results[check_name][-3:]  # Last 3 checks
            if not recent_results or not all(r["healthy"] for r in recent_results):
                return False

        return True


class AutoScaler:
    """Auto-scaling manager for deployments."""

    def __init__(self, deployment_provider: DeploymentProvider):
        self.provider = deployment_provider
        self.scaling_configs: Dict[str, ScalingConfig] = {}
        self.last_scale_time: Dict[str, datetime] = {}
        self.logger = logging.getLogger(__name__)

    def configure_scaling(self, deployment_id: str, config: ScalingConfig):
        """Configure scaling for a deployment."""
        self.scaling_configs[deployment_id] = config

    async def evaluate_scaling(
        self, deployment_id: str, metrics: Dict[str, float]
    ) -> Optional[int]:
        """Evaluate if scaling is needed."""
        if deployment_id not in self.scaling_configs:
            return None

        config = self.scaling_configs[deployment_id]

        # Check cooldown period
        if deployment_id in self.last_scale_time:
            time_since_last_scale = datetime.now() - self.last_scale_time[deployment_id]
            if time_since_last_scale < timedelta(seconds=config.cooldown_period):
                return None

        # Get current metrics
        current_metric = self._get_current_metric(config.strategy, metrics)
        if current_metric is None:
            return None

        # Get current replica count (simplified)
        current_replicas = await self._get_current_replicas(deployment_id)

        # Evaluate scaling decision
        if current_metric >= config.scale_up_threshold:
            new_replicas = min(current_replicas + 1, config.max_instances)
        elif current_metric <= config.scale_down_threshold:
            new_replicas = max(current_replicas - 1, config.min_instances)
        else:
            return None  # No scaling needed

        if new_replicas != current_replicas:
            self.last_scale_time[deployment_id] = datetime.now()
            return new_replicas

        return None

    def _get_current_metric(
        self, strategy: ScalingStrategy, metrics: Dict[str, float]
    ) -> Optional[float]:
        """Get the current metric value for scaling decisions."""
        if strategy == ScalingStrategy.CPU_BASED:
            return metrics.get("cpu_percent")
        elif strategy == ScalingStrategy.MEMORY_BASED:
            return metrics.get("memory_percent")
        elif strategy == ScalingStrategy.REQUEST_BASED:
            return metrics.get("request_queue_length")
        elif strategy == ScalingStrategy.CUSTOM_METRIC:
            return metrics.get("custom_metric")

        return None

    async def _get_current_replicas(self, deployment_id: str) -> int:
        """Get current replica count (simplified implementation)."""
        # In practice, this would query the deployment provider
        return 1  # Default assumption


class MetricsCollector:
    """System metrics collection."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def collect_system_metrics(self) -> Dict[str, float]:
        """Collect system-level metrics."""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage("/").percent,
            "network_connections": len(psutil.net_connections()),
        }

    def collect_process_metrics(self, pid: Optional[int] = None) -> Dict[str, float]:
        """Collect process-level metrics."""
        try:
            process = psutil.Process(pid) if pid else psutil.Process()
            memory_info = process.memory_info()

            return {
                "process_cpu_percent": process.cpu_percent(),
                "process_memory_rss": memory_info.rss,
                "process_memory_vms": memory_info.vms,
                "process_threads": process.num_threads(),
                "process_open_files": len(process.open_files()),
            }
        except psutil.NoSuchProcess:
            return {}

    def collect_application_metrics(
        self, app_context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Collect application-specific metrics."""
        # This would integrate with the UCUP monitoring system
        return {
            "active_agents": app_context.get("active_agents", 0),
            "pending_tasks": app_context.get("pending_tasks", 0),
            "completed_tasks": app_context.get("completed_tasks", 0),
            "average_confidence": app_context.get("avg_confidence", 0.0),
        }


class DeploymentManager:
    """High-level deployment manager coordinating all deployment activities."""

    def __init__(self, provider: DeploymentProvider = None):
        self.provider = provider or DockerDeploymentProvider()
        self.health_monitor = HealthMonitor()
        self.auto_scaler = AutoScaler(self.provider)
        self.metrics_collector = MetricsCollector()
        self.deployments: Dict[str, DeploymentSpec] = {}
        self.logger = logging.getLogger(__name__)

    async def deploy_ucup_system(
        self, config_path: Union[str, Path], provider_type: str = "docker"
    ) -> str:
        """Deploy a complete UCUP system from configuration."""
        from .config import load_ucup_config

        # Load configuration
        config = load_ucup_config(config_path)
        deployment_config = config.get("deployment", {})

        # Create deployment spec
        spec = DeploymentSpec(
            name=f"ucup-{deployment_config.get('environment', 'dev')}",
            environment=deployment_config.get("environment", "development"),
            container=ContainerConfig(
                image=deployment_config.get("container", {}).get(
                    "image", "ucup/framework:latest"
                ),
                ports=deployment_config.get("container", {}).get(
                    "ports", ["8000:8000"]
                ),
                environment=deployment_config.get("container", {}).get(
                    "environment", {}
                ),
            ),
            scaling=ScalingConfig(**deployment_config.get("scaling", {})),
        )

        # Add health checks
        health_config = deployment_config.get("health_checks", {})
        if health_config.get("enabled", True):
            spec.health_checks.append(
                HealthCheck(
                    name="health_endpoint",
                    endpoint="http://localhost:8000/health",
                    interval=health_config.get("interval", 30),
                    timeout=health_config.get("timeout", 10),
                    failure_threshold=health_config.get("failure_threshold", 3),
                )
            )

        # Deploy
        deployment_id = await self.provider.deploy(spec)
        self.deployments[deployment_id] = spec

        # Configure scaling
        self.auto_scaler.configure_scaling(deployment_id, spec.scaling)

        # Configure health monitoring
        for health_check in spec.health_checks:
            self.health_monitor.add_health_check(health_check)

        self.logger.info(f"Deployed UCUP system: {deployment_id}")
        return deployment_id

    async def monitor_system(self, deployment_id: str) -> Dict[str, Any]:
        """Monitor deployed system health and performance."""
        if deployment_id not in self.deployments:
            raise ValueError(f"Unknown deployment: {deployment_id}")

        # Run health checks
        health_results = await self.health_monitor.run_health_checks()

        # Collect metrics
        system_metrics = self.metrics_collector.collect_system_metrics()
        app_metrics = self.metrics_collector.collect_application_metrics({})

        # Evaluate auto-scaling
        combined_metrics = {**system_metrics, **app_metrics}
        scale_decision = await self.auto_scaler.evaluate_scaling(
            deployment_id, combined_metrics
        )

        if scale_decision:
            success = await self.provider.scale(deployment_id, scale_decision)
            if success:
                self.logger.info(f"Scaled {deployment_id} to {scale_decision} replicas")

        return {
            "deployment_id": deployment_id,
            "status": await self.provider.get_status(deployment_id),
            "health": health_results,
            "metrics": combined_metrics,
            "scaling_decision": scale_decision,
        }

    async def get_deployment_logs(self, deployment_id: str, lines: int = 100) -> str:
        """Get logs from deployment."""
        return await self.provider.get_logs(deployment_id, lines)

    async def undeploy_system(self, deployment_id: str) -> bool:
        """Remove deployed system."""
        success = await self.provider.undeploy(deployment_id)
        if success and deployment_id in self.deployments:
            del self.deployments[deployment_id]
        return success

    def list_deployments(self) -> List[str]:
        """List all active deployments."""
        return list(self.deployments.keys())


# Global deployment manager instance
_deployment_manager = None


def get_deployment_manager(provider: str = "docker") -> DeploymentManager:
    """Get the global deployment manager instance."""
    global _deployment_manager
    if _deployment_manager is None:
        if provider == "docker":
            deployment_provider = DockerDeploymentProvider()
        elif provider == "kubernetes":
            deployment_provider = KubernetesDeploymentProvider()
        else:
            raise ValueError(f"Unknown provider: {provider}")

        _deployment_manager = DeploymentManager(deployment_provider)

    return _deployment_manager
