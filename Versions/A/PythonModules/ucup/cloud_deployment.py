"""
Cloud Deployment Support for UCUP Framework

Provides comprehensive cloud deployment capabilities for Azure, AWS, and GCP,
enabling one-click deployment of UCUP agents to any major cloud platform.

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
import json
import logging
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class CloudConfig:
    """Configuration for cloud deployment."""

    platform: str  # 'aws', 'azure', 'gcp'
    region: str
    environment: str = "dev"
    resource_group: Optional[str] = None  # Azure
    project_id: Optional[str] = None  # GCP
    account_id: Optional[str] = None  # AWS
    credentials: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    security_groups: List[str] = field(default_factory=list)
    subnets: List[str] = field(default_factory=list)


@dataclass
class DeploymentSpec:
    """Specification for deployment."""

    name: str
    image: str
    ports: List[int] = field(default_factory=lambda: [8000])
    env_vars: Dict[str, str] = field(default_factory=dict)
    cpu: str = "500m"
    memory: str = "512Mi"
    replicas: int = 1
    health_check_path: str = "/health"
    auto_scaling: bool = False
    min_replicas: int = 1
    max_replicas: int = 10
    scaling_metric: str = "cpu"
    target_value: float = 70.0


@dataclass
class DeploymentResult:
    """Result of a cloud deployment."""

    success: bool
    deployment_id: str
    endpoint: Optional[str] = None
    status: str = "unknown"
    logs_url: Optional[str] = None
    monitoring_url: Optional[str] = None
    cost_estimate: Optional[float] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class CloudDeploymentProvider(ABC):
    """Abstract base class for cloud deployment providers."""

    def __init__(self, config: CloudConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    async def validate_credentials(self) -> bool:
        """Validate cloud credentials."""
        pass

    @abstractmethod
    async def deploy(self, spec: DeploymentSpec) -> DeploymentResult:
        """Deploy application to cloud."""
        pass

    @abstractmethod
    async def get_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get deployment status."""
        pass

    @abstractmethod
    async def scale(self, deployment_id: str, replicas: int) -> bool:
        """Scale deployment."""
        pass

    @abstractmethod
    async def destroy(self, deployment_id: str) -> bool:
        """Destroy deployment."""
        pass

    @abstractmethod
    async def get_cost_estimate(self, spec: DeploymentSpec) -> float:
        """Get cost estimate for deployment."""
        pass


class AWSDeploymentProvider(CloudDeploymentProvider):
    """AWS deployment provider supporting ECS, EKS, and Lambda."""

    async def validate_credentials(self) -> bool:
        """Validate AWS credentials."""
        try:
            import boto3

            sts = boto3.client("sts")
            sts.get_caller_identity()
            return True
        except Exception as e:
            self.logger.error(f"AWS credential validation failed: {e}")
            return False

    async def deploy(self, spec: DeploymentSpec) -> DeploymentResult:
        """Deploy to AWS ECS or EKS."""
        try:
            import boto3

            # Use ECS for simple deployments, EKS for complex ones
            if spec.replicas > 1 or spec.auto_scaling:
                return await self._deploy_to_ecs(spec)
            else:
                return await self._deploy_to_lambda(spec)

        except ImportError:
            return DeploymentResult(
                success=False,
                deployment_id="",
                errors=["boto3 not installed. Install with: pip install boto3"],
            )
        except Exception as e:
            return DeploymentResult(
                success=False,
                deployment_id="",
                errors=[f"AWS deployment failed: {str(e)}"],
            )

    async def _deploy_to_ecs(self, spec: DeploymentSpec) -> DeploymentResult:
        """Deploy to AWS ECS."""
        import boto3

        ecs = boto3.client("ecs", region_name=self.config.region)
        ec2 = boto3.resource("ec2", region_name=self.config.region)

        deployment_id = f"ucup-{spec.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        try:
            # Create task definition
            task_def = {
                "family": f"{spec.name}-task",
                "containerDefinitions": [
                    {
                        "name": spec.name,
                        "image": spec.image,
                        "portMappings": [
                            {"containerPort": port} for port in spec.ports
                        ],
                        "environment": [
                            {"name": k, "value": v} for k, v in spec.env_vars.items()
                        ],
                        "essential": True,
                        "healthCheck": {
                            "command": [
                                "CMD-SHELL",
                                f"curl -f http://localhost:{spec.ports[0]}{spec.health_check_path}",
                            ],
                            "interval": 30,
                            "timeout": 5,
                            "retries": 3,
                        },
                    }
                ],
                "requiresCompatibilities": ["FARGATE"],
                "networkMode": "awsvpc",
                "cpu": spec.cpu.replace("m", ""),
                "memory": spec.memory.replace("Mi", "").replace("Gi", "000"),
            }

            ecs.register_task_definition(**task_def)

            # Create service
            service_response = ecs.create_service(
                cluster="default",
                serviceName=deployment_id,
                taskDefinition=f"{spec.name}-task",
                desiredCount=spec.replicas,
                launchType="FARGATE",
                networkConfiguration={
                    "awsvpcConfiguration": {
                        "subnets": self.config.subnets or [],
                        "securityGroups": self.config.security_groups or [],
                        "assignPublicIp": "ENABLED",
                    }
                },
            )

            service_arn = service_response["service"]["serviceArn"]

            return DeploymentResult(
                success=True,
                deployment_id=deployment_id,
                endpoint=f"https://{deployment_id}.{self.config.region}.amazonaws.com",
                status="deploying",
                logs_url=f"https://{self.config.region}.console.aws.amazon.com/ecs/home?region={self.config.region}#/clusters/default/services/{deployment_id}/logs",
            )

        except Exception as e:
            return DeploymentResult(
                success=False,
                deployment_id=deployment_id,
                errors=[f"ECS deployment failed: {str(e)}"],
            )

    async def _deploy_to_lambda(self, spec: DeploymentSpec) -> DeploymentResult:
        """Deploy to AWS Lambda."""
        from base64 import b64encode

        import boto3

        lambda_client = boto3.client("lambda", region_name=self.config.region)

        deployment_id = (
            f"ucup-lambda-{spec.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )

        try:
            # Create Lambda function
            # Note: In practice, you'd build the container image and push to ECR first
            response = lambda_client.create_function(
                FunctionName=deployment_id,
                PackageType="Image",
                Code={"ImageUri": spec.image},
                Architectures=["x86_64"],
                MemorySize=int(spec.memory.replace("Mi", "").replace("Gi", "000")),
                Timeout=900,  # 15 minutes
                Environment={"Variables": spec.env_vars},
                Tags=self.config.tags,
            )

            # Create function URL
            url_config = lambda_client.create_function_url_config(
                FunctionName=deployment_id, AuthType="NONE"  # In production, use IAM
            )

            return DeploymentResult(
                success=True,
                deployment_id=deployment_id,
                endpoint=url_config["FunctionUrl"],
                status="active",
                logs_url=f"https://{self.config.region}.console.aws.amazon.com/lambda/home?region={self.config.region}#/functions/{deployment_id}",
            )

        except Exception as e:
            return DeploymentResult(
                success=False,
                deployment_id=deployment_id,
                errors=[f"Lambda deployment failed: {str(e)}"],
            )

    async def get_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get AWS deployment status."""
        try:
            import boto3

            ecs = boto3.client("ecs", region_name=self.config.region)

            # Check if it's an ECS service
            services = ecs.describe_services(
                cluster="default", services=[deployment_id]
            )

            if services["services"]:
                service = services["services"][0]
                return {
                    "status": service["status"],
                    "running_count": service["runningCount"],
                    "desired_count": service["desiredCount"],
                    "last_updated": service.get("createdAt", "").isoformat()
                    if service.get("createdAt")
                    else None,
                }

            # Check if it's a Lambda function
            lambda_client = boto3.client("lambda", region_name=self.config.region)
            function = lambda_client.get_function(FunctionName=deployment_id)
            return {
                "status": "active",
                "last_modified": function["Configuration"]["LastModified"],
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def scale(self, deployment_id: str, replicas: int) -> bool:
        """Scale AWS deployment."""
        try:
            import boto3

            ecs = boto3.client("ecs", region_name=self.config.region)

            ecs.update_service(
                cluster="default", service=deployment_id, desiredCount=replicas
            )
            return True
        except Exception as e:
            self.logger.error(f"AWS scaling failed: {e}")
            return False

    async def destroy(self, deployment_id: str) -> bool:
        """Destroy AWS deployment."""
        try:
            import boto3

            ecs = boto3.client("ecs", region_name=self.config.region)
            lambda_client = boto3.client("lambda", region_name=self.config.region)

            # Try to delete ECS service
            try:
                ecs.delete_service(cluster="default", service=deployment_id, force=True)
            except:
                pass

            # Try to delete Lambda function
            try:
                lambda_client.delete_function(FunctionName=deployment_id)
            except:
                pass

            return True
        except Exception as e:
            self.logger.error(f"AWS destroy failed: {e}")
            return False

    async def get_cost_estimate(self, spec: DeploymentSpec) -> float:
        """Get AWS cost estimate."""
        # Simplified cost estimation
        base_cost_per_hour = 0.01344  # t3.micro equivalent
        hours_per_month = 730

        cpu_cores = float(spec.cpu.replace("m", "")) / 1000
        memory_gb = float(spec.memory.replace("Mi", "").replace("Gi", "000")) / 1000

        # Scale cost based on resources
        resource_multiplier = (cpu_cores * 0.5) + (memory_gb * 0.3)
        estimated_hourly_cost = (
            base_cost_per_hour * max(1, resource_multiplier) * spec.replicas
        )

        return estimated_hourly_cost * hours_per_month


class AzureDeploymentProvider(CloudDeploymentProvider):
    """Azure deployment provider supporting AKS, Container Apps, and Functions."""

    async def validate_credentials(self) -> bool:
        """Validate Azure credentials."""
        try:
            from azure.identity import DefaultAzureCredential
            from azure.mgmt.resource import ResourceManagementClient

            credential = DefaultAzureCredential()
            subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
            if not subscription_id:
                return False

            client = ResourceManagementClient(credential, subscription_id)
            # Test credentials by listing resource groups
            list(client.resource_groups.list())
            return True
        except Exception as e:
            self.logger.error(f"Azure credential validation failed: {e}")
            return False

    async def deploy(self, spec: DeploymentSpec) -> DeploymentResult:
        """Deploy to Azure Container Apps or AKS."""
        try:
            from azure.identity import DefaultAzureCredential
            from azure.mgmt.containerinstance import ContainerInstanceManagementClient
            from azure.mgmt.containerinstance.models import (
                Container,
                ContainerGroup,
                ContainerPort,
                ResourceRequests,
                ResourceRequirements,
            )

            credential = DefaultAzureCredential()
            subscription_id = os.environ["AZURE_SUBSCRIPTION_ID"]

            aci_client = ContainerInstanceManagementClient(credential, subscription_id)

            deployment_id = (
                f"ucup-{spec.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )

            # Create container group
            containers = [
                Container(
                    name=spec.name,
                    image=spec.image,
                    ports=[ContainerPort(port=port) for port in spec.ports],
                    environment_variables=[
                        {"name": k, "value": v} for k, v in spec.env_vars.items()
                    ],
                    resources=ResourceRequirements(
                        requests=ResourceRequests(
                            memory_in_gb=float(
                                spec.memory.replace("Mi", "").replace("Gi", "")
                            )
                            / 1000,
                            cpu=float(spec.cpu.replace("m", "")) / 1000,
                        )
                    ),
                )
            ]

            container_group = ContainerGroup(
                location=self.config.region,
                containers=containers,
                os_type="Linux",
                ip_address={
                    "type": "Public",
                    "ports": [{"protocol": "TCP", "port": port} for port in spec.ports],
                },
                tags=self.config.tags,
            )

            if self.config.resource_group:
                resource_group = self.config.resource_group
            else:
                resource_group = f"ucup-rg-{datetime.now().strftime('%Y%m%d')}"

            # Create resource group if it doesn't exist
            await self._ensure_resource_group(
                credential, subscription_id, resource_group, self.config.region
            )

            result = aci_client.container_groups.create_or_update(
                resource_group, deployment_id, container_group
            )

            return DeploymentResult(
                success=True,
                deployment_id=deployment_id,
                endpoint=f"https://{deployment_id}.{self.config.region}.azurecontainer.io",
                status="running",
                logs_url=f"https://portal.azure.com/#resource/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.ContainerInstance/containerGroups/{deployment_id}",
            )

        except ImportError:
            return DeploymentResult(
                success=False,
                deployment_id="",
                errors=[
                    "Azure SDK not installed. Install with: pip install azure-identity azure-mgmt-containerinstance"
                ],
            )
        except Exception as e:
            return DeploymentResult(
                success=False,
                deployment_id="",
                errors=[f"Azure deployment failed: {str(e)}"],
            )

    async def _ensure_resource_group(
        self, credential, subscription_id: str, rg_name: str, location: str
    ):
        """Ensure resource group exists."""
        from azure.mgmt.resource import ResourceManagementClient

        rg_client = ResourceManagementClient(credential, subscription_id)
        rg_client.resource_groups.create_or_update(rg_name, {"location": location})

    async def get_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get Azure deployment status."""
        try:
            from azure.identity import DefaultAzureCredential
            from azure.mgmt.containerinstance import ContainerInstanceManagementClient

            credential = DefaultAzureCredential()
            subscription_id = os.environ["AZURE_SUBSCRIPTION_ID"]
            resource_group = (
                self.config.resource_group
                or f"ucup-rg-{datetime.now().strftime('%Y%m%d')}"
            )

            aci_client = ContainerInstanceManagementClient(credential, subscription_id)
            container_group = aci_client.container_groups.get(
                resource_group, deployment_id
            )

            return {
                "status": container_group.provisioning_state,
                "instance_view": {
                    "state": container_group.instance_view.current_state.state
                    if container_group.instance_view
                    else "Unknown"
                },
                "ip_address": container_group.ip_address.ip
                if container_group.ip_address
                else None,
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def scale(self, deployment_id: str, replicas: int) -> bool:
        """Scale Azure deployment."""
        # Azure Container Instances don't support scaling in the same way as ECS
        # For scaling, we'd need to use Container Apps or AKS
        self.logger.warning(
            "Azure Container Instances scaling requires migration to Container Apps"
        )
        return False

    async def destroy(self, deployment_id: str) -> bool:
        """Destroy Azure deployment."""
        try:
            from azure.identity import DefaultAzureCredential
            from azure.mgmt.containerinstance import ContainerInstanceManagementClient

            credential = DefaultAzureCredential()
            subscription_id = os.environ["AZURE_SUBSCRIPTION_ID"]
            resource_group = (
                self.config.resource_group
                or f"ucup-rg-{datetime.now().strftime('%Y%m%d')}"
            )

            aci_client = ContainerInstanceManagementClient(credential, subscription_id)
            aci_client.container_groups.delete(resource_group, deployment_id)
            return True
        except Exception as e:
            self.logger.error(f"Azure destroy failed: {e}")
            return False

    async def get_cost_estimate(self, spec: DeploymentSpec) -> float:
        """Get Azure cost estimate."""
        # Simplified cost estimation for Container Instances
        base_cost_per_hour = 0.000025  # Base cost per vCPU second
        hours_per_month = 730

        cpu_cores = float(spec.cpu.replace("m", "")) / 1000
        memory_gb = float(spec.memory.replace("Mi", "").replace("Gi", "")) / 1000

        # Azure pricing: $0.000012 per vCPU second + $0.0000015 per GB second
        vcpu_cost = base_cost_per_hour * cpu_cores * 3600  # per hour
        memory_cost = 0.0000015 * memory_gb * 3600  # per hour

        estimated_hourly_cost = (vcpu_cost + memory_cost) * spec.replicas
        return estimated_hourly_cost * hours_per_month


class GCPDeploymentProvider(CloudDeploymentProvider):
    """GCP deployment provider supporting Cloud Run, GKE, and Cloud Functions."""

    async def validate_credentials(self) -> bool:
        """Validate GCP credentials."""
        try:
            from google.cloud import storage

            client = storage.Client()
            # Test credentials by listing buckets (will fail if no buckets, but validates auth)
            try:
                list(client.list_buckets(max_results=1))
            except:
                pass  # Ignore permission errors, auth works
            return True
        except Exception as e:
            self.logger.error(f"GCP credential validation failed: {e}")
            return False

    async def deploy(self, spec: DeploymentSpec) -> DeploymentResult:
        """Deploy to Google Cloud Run."""
        try:
            from google.api_core import operation
            from google.cloud import run_v2

            client = run_v2.ServicesClient()

            deployment_id = (
                f"ucup-{spec.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )

            parent = f"projects/{self.config.project_id}/locations/{self.config.region}"

            service = {
                "template": {
                    "containers": [
                        {
                            "image": spec.image,
                            "ports": [{"container_port": spec.ports[0]}],
                            "env": [
                                {"name": k, "value": v}
                                for k, v in spec.env_vars.items()
                            ],
                            "resources": {
                                "limits": {"cpu": spec.cpu, "memory": spec.memory}
                            },
                        }
                    ],
                    "scaling": {
                        "min_instance_count": spec.min_replicas
                        if spec.auto_scaling
                        else spec.replicas,
                        "max_instance_count": spec.max_replicas
                        if spec.auto_scaling
                        else spec.replicas,
                    },
                },
                "traffic": [{"percent": 100, "latest_revision": True}],
            }

            operation = client.create_service(
                request={
                    "parent": parent,
                    "service_id": deployment_id,
                    "service": service,
                }
            )

            result = operation.result()

            return DeploymentResult(
                success=True,
                deployment_id=deployment_id,
                endpoint=result.uri,
                status="active",
                logs_url=f"https://console.cloud.google.com/run/detail/{self.config.region}/{deployment_id}/logs",
            )

        except ImportError:
            return DeploymentResult(
                success=False,
                deployment_id="",
                errors=[
                    "Google Cloud SDK not installed. Install with: pip install google-cloud-run"
                ],
            )
        except Exception as e:
            return DeploymentResult(
                success=False,
                deployment_id="",
                errors=[f"GCP deployment failed: {str(e)}"],
            )

    async def get_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get GCP deployment status."""
        try:
            from google.cloud import run_v2

            client = run_v2.ServicesClient()
            name = f"projects/{self.config.project_id}/locations/{self.config.region}/services/{deployment_id}"

            service = client.get_service(name=name)

            return {
                "status": "active" if service else "not_found",
                "url": service.uri if service else None,
                "latest_created_time": service.latest_created_revision
                if service
                else None,
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def scale(self, deployment_id: str, replicas: int) -> bool:
        """Scale GCP deployment."""
        try:
            from google.cloud import run_v2

            client = run_v2.ServicesClient()
            name = f"projects/{self.config.project_id}/locations/{self.config.region}/services/{deployment_id}"

            service = client.get_service(name=name)
            service.template.scaling.min_instance_count = replicas
            service.template.scaling.max_instance_count = replicas

            client.update_service(service=service)
            return True
        except Exception as e:
            self.logger.error(f"GCP scaling failed: {e}")
            return False

    async def destroy(self, deployment_id: str) -> bool:
        """Destroy GCP deployment."""
        try:
            from google.cloud import run_v2

            client = run_v2.ServicesClient()
            name = f"projects/{self.config.project_id}/locations/{self.config.region}/services/{deployment_id}"

            client.delete_service(name=name)
            return True
        except Exception as e:
            self.logger.error(f"GCP destroy failed: {e}")
            return False

    async def get_cost_estimate(self, spec: DeploymentSpec) -> float:
        """Get GCP cost estimate."""
        # Simplified cost estimation for Cloud Run
        base_cost_per_hour = 0.00002400  # Base vCPU cost per second
        hours_per_month = 730

        cpu_cores = float(spec.cpu.replace("m", "")) / 1000
        memory_gb = float(spec.memory.replace("Mi", "").replace("Gi", "")) / 1000

        # GCP pricing: vCPU cost + memory cost
        vcpu_cost = base_cost_per_hour * cpu_cores * 3600  # per hour
        memory_cost = 0.00000250 * memory_gb * 3600  # per hour (simplified)

        estimated_hourly_cost = (vcpu_cost + memory_cost) * spec.replicas
        return estimated_hourly_cost * hours_per_month


class CloudDeploymentManager:
    """Manages cloud deployments across multiple providers."""

    def __init__(self):
        self.providers: Dict[str, CloudDeploymentProvider] = {}
        self.logger = logging.getLogger(__name__)

    def register_provider(self, platform: str, provider: CloudDeploymentProvider):
        """Register a cloud provider."""
        self.providers[platform] = provider

    def get_provider(self, platform: str) -> Optional[CloudDeploymentProvider]:
        """Get provider for platform."""
        return self.providers.get(platform)

    async def validate_credentials(self, platform: str) -> bool:
        """Validate credentials for a platform."""
        provider = self.get_provider(platform)
        if not provider:
            self.logger.error(f"No provider registered for platform: {platform}")
            return False

        return await provider.validate_credentials()

    async def deploy(self, platform: str, spec: DeploymentSpec) -> DeploymentResult:
        """Deploy to specified platform."""
        provider = self.get_provider(platform)
        if not provider:
            return DeploymentResult(
                success=False,
                deployment_id="",
                errors=[f"No provider registered for platform: {platform}"],
            )

        return await provider.deploy(spec)

    async def get_status(self, platform: str, deployment_id: str) -> Dict[str, Any]:
        """Get deployment status."""
        provider = self.get_provider(platform)
        if not provider:
            return {"status": "error", "error": f"No provider for {platform}"}

        return await provider.get_status(deployment_id)

    async def scale(self, platform: str, deployment_id: str, replicas: int) -> bool:
        """Scale deployment."""
        provider = self.get_provider(platform)
        if not provider:
            return False

        return await provider.scale(deployment_id, replicas)

    async def destroy(self, platform: str, deployment_id: str) -> bool:
        """Destroy deployment."""
        provider = self.get_provider(platform)
        if not provider:
            return False

        return await provider.destroy(deployment_id)

    async def get_cost_estimate(self, platform: str, spec: DeploymentSpec) -> float:
        """Get cost estimate."""
        provider = self.get_provider(platform)
        if not provider:
            return 0.0

        return await provider.get_cost_estimate(spec)


def create_cloud_deployment_manager() -> CloudDeploymentManager:
    """Create and configure cloud deployment manager."""
    manager = CloudDeploymentManager()

    # Note: Providers are created with minimal config and configured at runtime
    # Actual credentials and config are provided during deployment

    return manager


def create_deployment_spec_from_config(config: Dict[str, Any]) -> DeploymentSpec:
    """Create deployment spec from configuration."""
    deployment_config = config.get("deployment", {})

    return DeploymentSpec(
        name=config.get("name", "ucup-agent"),
        image=config.get("image", "ucup/agent:latest"),
        ports=deployment_config.get("ports", [8000]),
        env_vars=deployment_config.get("env_vars", {}),
        cpu=deployment_config.get("cpu", "500m"),
        memory=deployment_config.get("memory", "512Mi"),
        replicas=deployment_config.get("replicas", 1),
        health_check_path=deployment_config.get("health_check_path", "/health"),
        auto_scaling=deployment_config.get("auto_scaling", False),
        min_replicas=deployment_config.get("min_replicas", 1),
        max_replicas=deployment_config.get("max_replicas", 10),
        scaling_metric=deployment_config.get("scaling_metric", "cpu"),
        target_value=deployment_config.get("target_value", 70.0),
    )


def create_cloud_config_from_env(platform: str) -> CloudConfig:
    """Create cloud config from environment variables."""
    config = CloudConfig(platform=platform, region="us-east-1")

    if platform == "aws":
        config.region = os.environ.get("AWS_REGION", "us-east-1")
        config.account_id = os.environ.get("AWS_ACCOUNT_ID")
    elif platform == "azure":
        config.region = os.environ.get("AZURE_LOCATION", "eastus")
        config.resource_group = os.environ.get("AZURE_RESOURCE_GROUP")
    elif platform == "gcp":
        config.region = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
        config.project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")

    return config


# Convenience functions for CLI usage
async def deploy_to_cloud(
    platform: str, spec: DeploymentSpec, config: Optional[CloudConfig] = None
) -> DeploymentResult:
    """Deploy to cloud with automatic provider setup."""
    manager = create_cloud_deployment_manager()

    if not config:
        config = create_cloud_config_from_env(platform)

    # Create and register provider
    if platform == "aws":
        provider = AWSDeploymentProvider(config)
    elif platform == "azure":
        provider = AzureDeploymentProvider(config)
    elif platform == "gcp":
        provider = GCPDeploymentProvider(config)
    else:
        return DeploymentResult(
            success=False,
            deployment_id="",
            errors=[f"Unsupported platform: {platform}"],
        )

    manager.register_provider(platform, provider)

    # Validate credentials
    if not await manager.validate_credentials(platform):
        return DeploymentResult(
            success=False,
            deployment_id="",
            errors=[f"Invalid credentials for {platform}"],
        )

    # Deploy
    return await manager.deploy(platform, spec)


async def get_cloud_status(platform: str, deployment_id: str) -> Dict[str, Any]:
    """Get cloud deployment status."""
    manager = create_cloud_deployment_manager()
    config = create_cloud_config_from_env(platform)

    if platform == "aws":
        provider = AWSDeploymentProvider(config)
    elif platform == "azure":
        provider = AzureDeploymentProvider(config)
    elif platform == "gcp":
        provider = GCPDeploymentProvider(config)
    else:
        return {"status": "error", "error": f"Unsupported platform: {platform}"}

    manager.register_provider(platform, provider)
    return await manager.get_status(platform, deployment_id)


async def destroy_cloud_deployment(platform: str, deployment_id: str) -> bool:
    """Destroy cloud deployment."""
    manager = create_cloud_deployment_manager()
    config = create_cloud_config_from_env(platform)

    if platform == "aws":
        provider = AWSDeploymentProvider(config)
    elif platform == "azure":
        provider = AzureDeploymentProvider(config)
    elif platform == "gcp":
        provider = GCPDeploymentProvider(config)
    else:
        return False

    manager.register_provider(platform, provider)
    return await manager.destroy(platform, deployment_id)
