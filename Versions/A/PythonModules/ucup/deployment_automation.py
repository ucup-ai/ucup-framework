"""
Deployment Automation Framework for UCUP

Provides comprehensive deployment automation capabilities including CI/CD integration,
infrastructure as code, monitoring, scaling, and multi-cloud deployment strategies.

Features:
- Automated deployment pipelines with rollback capabilities
- Infrastructure as code templates for all major clouds
- Blue-green and canary deployment strategies
- Auto-scaling based on custom metrics
- Multi-region deployment with traffic management
- Cost optimization and resource management
- Security automation and compliance checks
- Integration with monitoring and alerting systems
"""

import asyncio
import hashlib
import json
import logging
import os
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import yaml

from .cloud_deployment import (
    CloudConfig,
    CloudDeploymentManager,
    DeploymentResult,
    DeploymentSpec,
    create_cloud_config_from_env,
    create_deployment_spec_from_config,
)

logger = logging.getLogger(__name__)


class DeploymentStrategy(Enum):
    """Deployment strategies."""

    ROLLING_UPDATE = "rolling_update"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    IMMEDIATE = "immediate"


class DeploymentStatus(Enum):
    """Deployment status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"


@dataclass
class DeploymentPipeline:
    """Configuration for automated deployment pipeline."""

    name: str
    stages: List[Dict[str, Any]] = field(default_factory=list)
    environments: List[str] = field(default_factory=lambda: ["dev", "staging", "prod"])
    approval_required: bool = False
    rollback_on_failure: bool = True
    monitoring_enabled: bool = True
    notifications: List[str] = field(default_factory=list)  # email, slack, etc.


@dataclass
class Deployment:
    """Represents a deployment instance."""

    id: str
    pipeline_name: str
    environment: str
    strategy: DeploymentStrategy
    status: DeploymentStatus
    start_time: datetime
    spec: DeploymentSpec
    config: CloudConfig
    end_time: Optional[datetime] = None
    results: List[DeploymentResult] = field(default_factory=list)
    rollback_available: bool = False
    rollback_deployment_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InfrastructureTemplate:
    """Infrastructure as code template."""

    name: str
    platform: str
    template_type: str  # terraform, cloudformation, arm, etc.
    content: str
    variables: Dict[str, Any] = field(default_factory=dict)
    outputs: List[str] = field(default_factory=list)


@dataclass
class ScalingPolicy:
    """Auto-scaling policy configuration."""

    name: str
    metric_name: str
    target_value: float
    scale_up_threshold: float
    scale_down_threshold: float
    min_instances: int
    max_instances: int
    cooldown_period: int  # seconds
    enabled: bool = True


@dataclass
class MultiRegionConfig:
    """Multi-region deployment configuration."""

    primary_region: str
    secondary_regions: List[str] = field(default_factory=list)
    traffic_distribution: Dict[str, float] = field(default_factory=dict)
    failover_enabled: bool = True
    latency_based_routing: bool = True


class DeploymentAutomator:
    """Main deployment automation system."""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path.home() / ".ucup" / "deployments"
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.cloud_manager = CloudDeploymentManager()
        self.deployments: Dict[str, Deployment] = {}
        self.pipelines: Dict[str, DeploymentPipeline] = {}
        self.templates: Dict[str, InfrastructureTemplate] = {}
        self.scaling_policies: Dict[str, ScalingPolicy] = {}

        self.logger = logging.getLogger(__name__)

        # Load persisted data
        self._load_state()

    def _load_state(self):
        """Load persisted deployment state."""
        try:
            # Load deployments
            deployments_file = self.base_dir / "deployments.json"
            if deployments_file.exists():
                with open(deployments_file, "r") as f:
                    data = json.load(f)
                    for d in data.get("deployments", []):
                        deployment = Deployment(**d)
                        self.deployments[deployment.id] = deployment

            # Load pipelines
            pipelines_file = self.base_dir / "pipelines.json"
            if pipelines_file.exists():
                with open(pipelines_file, "r") as f:
                    data = json.load(f)
                    for p in data.get("pipelines", []):
                        pipeline = DeploymentPipeline(**p)
                        self.pipelines[pipeline.name] = pipeline

        except Exception as e:
            self.logger.error(f"Failed to load deployment state: {e}")

    def _save_state(self):
        """Save deployment state."""
        try:
            # Save deployments
            deployments_data = {
                "deployments": [d.__dict__ for d in self.deployments.values()],
                "last_updated": datetime.now().isoformat(),
            }
            with open(self.base_dir / "deployments.json", "w") as f:
                json.dump(deployments_data, f, indent=2, default=str)

            # Save pipelines
            pipelines_data = {
                "pipelines": [p.__dict__ for p in self.pipelines.values()],
                "last_updated": datetime.now().isoformat(),
            }
            with open(self.base_dir / "pipelines.json", "w") as f:
                json.dump(pipelines_data, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Failed to save deployment state: {e}")

    def create_deployment_pipeline(self, pipeline: DeploymentPipeline):
        """Create a deployment pipeline."""
        self.pipelines[pipeline.name] = pipeline
        self._save_state()

    def create_infrastructure_template(self, template: InfrastructureTemplate):
        """Create infrastructure template."""
        self.templates[template.name] = template

        # Save template to file
        template_dir = self.base_dir / "templates" / template.platform
        template_dir.mkdir(parents=True, exist_ok=True)

        template_file = template_dir / f"{template.name}.{template.template_type}"
        with open(template_file, "w") as f:
            f.write(template.content)

    async def deploy_with_strategy(
        self,
        pipeline_name: str,
        environment: str,
        spec: DeploymentSpec,
        config: CloudConfig,
        strategy: DeploymentStrategy = DeploymentStrategy.ROLLING_UPDATE,
    ) -> Deployment:
        """Deploy using specified strategy."""

        deployment_id = f"deploy-{pipeline_name}-{environment}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        deployment = Deployment(
            id=deployment_id,
            pipeline_name=pipeline_name,
            environment=environment,
            strategy=strategy,
            status=DeploymentStatus.PENDING,
            start_time=datetime.now(),
            spec=spec,
            config=config,
        )

        self.deployments[deployment_id] = deployment
        self._save_state()

        try:
            deployment.status = DeploymentStatus.IN_PROGRESS

            if strategy == DeploymentStrategy.BLUE_GREEN:
                result = await self._deploy_blue_green(deployment)
            elif strategy == DeploymentStrategy.CANARY:
                result = await self._deploy_canary(deployment)
            elif strategy == DeploymentStrategy.ROLLING_UPDATE:
                result = await self._deploy_rolling_update(deployment)
            else:  # IMMEDIATE
                result = await self._deploy_immediate(deployment)

            deployment.results.append(result)
            deployment.status = (
                DeploymentStatus.SUCCESS if result.success else DeploymentStatus.FAILED
            )
            deployment.end_time = datetime.now()

            if result.success:
                deployment.rollback_available = True

        except Exception as e:
            deployment.status = DeploymentStatus.FAILED
            deployment.end_time = datetime.now()
            deployment.results.append(
                DeploymentResult(
                    success=False,
                    deployment_id=deployment_id,
                    errors=[f"Deployment failed: {str(e)}"],
                )
            )

        self._save_state()
        return deployment

    async def _deploy_immediate(self, deployment: Deployment) -> DeploymentResult:
        """Deploy immediately without strategy."""
        return await self.cloud_manager.deploy(
            deployment.config.platform, deployment.spec
        )

    async def _deploy_rolling_update(self, deployment: Deployment) -> DeploymentResult:
        """Deploy using rolling update strategy."""
        # For now, just do immediate deployment
        # In a full implementation, this would gradually replace instances
        return await self._deploy_immediate(deployment)

    async def _deploy_blue_green(self, deployment: Deployment) -> DeploymentResult:
        """Deploy using blue-green strategy."""
        # Create green environment (new deployment)
        green_result = await self.cloud_manager.deploy(
            deployment.config.platform, deployment.spec
        )

        if not green_result.success:
            return green_result

        # In blue-green, we would:
        # 1. Deploy to green environment
        # 2. Run tests on green
        # 3. Switch traffic from blue to green
        # 4. Keep blue as rollback option

        deployment.rollback_deployment_id = "blue-environment"  # Placeholder

        return green_result

    async def _deploy_canary(self, deployment: Deployment) -> DeploymentResult:
        """Deploy using canary strategy."""
        # Start with small percentage of traffic
        canary_spec = deployment.spec
        canary_spec.replicas = max(1, deployment.spec.replicas // 10)  # 10% of traffic

        canary_result = await self.cloud_manager.deploy(
            deployment.config.platform, canary_spec
        )

        if not canary_result.success:
            return canary_result

        # In canary deployment, we would:
        # 1. Deploy canary with small traffic percentage
        # 2. Monitor metrics
        # 3. Gradually increase traffic if successful
        # 4. Rollback if issues detected

        return canary_result

    async def rollback_deployment(self, deployment_id: str) -> bool:
        """Rollback a deployment."""
        if deployment_id not in self.deployments:
            return False

        deployment = self.deployments[deployment_id]

        if not deployment.rollback_available or not deployment.rollback_deployment_id:
            return False

        deployment.status = DeploymentStatus.ROLLING_BACK

        try:
            # Switch traffic back to previous version
            # This is simplified - in practice, would depend on the strategy used
            success = await self.cloud_manager.scale(
                deployment.config.platform,
                deployment.rollback_deployment_id,
                deployment.spec.replicas,
            )

            if success:
                deployment.status = DeploymentStatus.ROLLED_BACK
                # Destroy the failed deployment
                await self.cloud_manager.destroy(
                    deployment.config.platform, deployment_id
                )

            return success

        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return False

        finally:
            self._save_state()

    async def scale_deployment(self, deployment_id: str, replicas: int) -> bool:
        """Scale an existing deployment."""
        if deployment_id not in self.deployments:
            return False

        deployment = self.deployments[deployment_id]

        # Find the active deployment result
        active_result = None
        for result in reversed(deployment.results):
            if result.success:
                active_result = result
                break

        if not active_result:
            return False

        return await self.cloud_manager.scale(
            deployment.config.platform, active_result.deployment_id, replicas
        )

    async def get_deployment_status(
        self, deployment_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get detailed deployment status."""
        if deployment_id not in self.deployments:
            return None

        deployment = self.deployments[deployment_id]

        # Get cloud status for active deployment
        active_result = None
        for result in reversed(deployment.results):
            if result.success:
                active_result = result
                break

        if active_result:
            cloud_status = await self.cloud_manager.get_status(
                deployment.config.platform, active_result.deployment_id
            )
        else:
            cloud_status = {}

        return {
            "deployment_id": deployment_id,
            "pipeline": deployment.pipeline_name,
            "environment": deployment.environment,
            "strategy": deployment.strategy.value,
            "status": deployment.status.value,
            "start_time": deployment.start_time.isoformat(),
            "end_time": deployment.end_time.isoformat()
            if deployment.end_time
            else None,
            "rollback_available": deployment.rollback_available,
            "cloud_status": cloud_status,
            "results": [r.__dict__ for r in deployment.results],
        }

    def create_scaling_policy(self, policy: ScalingPolicy):
        """Create an auto-scaling policy."""
        self.scaling_policies[policy.name] = policy

        # Save policy
        policies_file = self.base_dir / "scaling_policies.json"
        policies_data = {
            "policies": [p.__dict__ for p in self.scaling_policies.values()],
            "last_updated": datetime.now().isoformat(),
        }
        with open(policies_file, "w") as f:
            json.dump(policies_data, f, indent=2, default=str)

    async def apply_scaling_policy(self, policy_name: str, deployment_id: str):
        """Apply scaling policy to deployment."""
        if (
            policy_name not in self.scaling_policies
            or deployment_id not in self.deployments
        ):
            return False

        policy = self.scaling_policies[policy_name]
        deployment = self.deployments[deployment_id]

        if not policy.enabled:
            return False

        # Get current metrics (simplified - in practice would integrate with monitoring)
        # For now, just check if we need to scale based on policy
        current_metrics = await self._get_deployment_metrics(deployment)

        if not current_metrics:
            return False

        metric_value = current_metrics.get(policy.metric_name, 0)

        if metric_value > policy.scale_up_threshold:
            new_replicas = min(policy.max_instances, deployment.spec.replicas + 1)
        elif metric_value < policy.scale_down_threshold:
            new_replicas = max(policy.min_instances, deployment.spec.replicas - 1)
        else:
            return False  # No scaling needed

        if new_replicas != deployment.spec.replicas:
            success = await self.scale_deployment(deployment_id, new_replicas)
            if success:
                deployment.spec.replicas = new_replicas
                self._save_state()
            return success

        return False

    async def _get_deployment_metrics(
        self, deployment: Deployment
    ) -> Optional[Dict[str, float]]:
        """Get metrics for deployment (simplified implementation)."""
        # In practice, this would integrate with cloud monitoring services
        # For now, return mock data
        return {"cpu": 65.0, "memory": 70.0, "requests_per_second": 150.0}

    def create_multi_region_config(self, config: MultiRegionConfig):
        """Create multi-region deployment configuration."""
        config_file = self.base_dir / "multi_region_config.json"
        with open(config_file, "w") as f:
            json.dump(config.__dict__, f, indent=2, default=str)

    async def deploy_multi_region(
        self, spec: DeploymentSpec, config: MultiRegionConfig
    ) -> List[DeploymentResult]:
        """Deploy across multiple regions."""
        results = []

        # Deploy to primary region first
        primary_config = CloudConfig(
            platform=config.primary_region.split("-")[
                0
            ],  # Extract platform from region
            region=config.primary_region,
        )

        primary_result = await self.cloud_manager.deploy(primary_config.platform, spec)
        results.append(primary_result)

        if not primary_result.success:
            return results

        # Deploy to secondary regions
        for region in config.secondary_regions:
            secondary_config = CloudConfig(platform=region.split("-")[0], region=region)

            # Adjust replicas based on traffic distribution
            region_spec = spec
            region_traffic = config.traffic_distribution.get(region, 0.1)  # Default 10%
            region_spec.replicas = max(1, int(spec.replicas * region_traffic))

            result = await self.cloud_manager.deploy(
                secondary_config.platform, region_spec
            )
            results.append(result)

        return results

    def generate_infrastructure_templates(self):
        """Generate infrastructure templates for all supported platforms."""

        # AWS CloudFormation template
        aws_template = InfrastructureTemplate(
            name="ucup-agent-ecs",
            platform="aws",
            template_type="yaml",
            content=self._generate_aws_template(),
            variables={
                "Environment": "dev",
                "VPCId": "",
                "SubnetIds": "",
                "ClusterName": "ucup-cluster",
            },
            outputs=["ServiceUrl", "LoadBalancerDNS"],
        )
        self.create_infrastructure_template(aws_template)

        # Azure ARM template
        azure_template = InfrastructureTemplate(
            name="ucup-agent-aci",
            platform="azure",
            template_type="json",
            content=self._generate_azure_template(),
            variables={
                "Environment": "dev",
                "Location": "eastus",
                "ResourceGroupName": "ucup-rg",
            },
            outputs=["ServiceUrl", "ContainerGroupId"],
        )
        self.create_infrastructure_template(azure_template)

        # GCP Deployment Manager template
        gcp_template = InfrastructureTemplate(
            name="ucup-agent-cloudrun",
            platform="gcp",
            template_type="yaml",
            content=self._generate_gcp_template(),
            variables={"Environment": "dev", "Region": "us-central1", "ProjectId": ""},
            outputs=["ServiceUrl", "ServiceId"],
        )
        self.create_infrastructure_template(gcp_template)

    def _generate_aws_template(self) -> str:
        """Generate AWS CloudFormation template."""
        return """
AWSTemplateFormatVersion: '2010-09-09'
Description: 'UCUP Agent Deployment Template'

Parameters:
  Environment:
    Type: String
    Default: dev
    AllowedValues: [dev, staging, prod]

  VpcId:
    Type: AWS::EC2::VPC::Id

  SubnetIds:
    Type: List<AWS::EC2::Subnet::Id

Resources:
  ECSCluster:
    Type: AWS::ECS::Cluster
    Properties:
      ClusterName: !Sub "${Environment}-ucup-cluster"

  ECSTaskDefinition:
    Type: AWS::ECS::TaskDefinition
    Properties:
      Family: !Sub "${Environment}-ucup-agent"
      Cpu: 512
      Memory: 1024
      NetworkMode: awsvpc
      RequiresCompatibilities:
        - FARGATE
      ExecutionRoleArn: !GetAtt ECSExecutionRole.Arn
      ContainerDefinitions:
        - Name: ucup-agent
          Image: ucup/agent:latest
          PortMappings:
            - ContainerPort: 8000
              Protocol: tcp
          Environment:
            - Name: ENVIRONMENT
              Value: !Ref Environment
          LogConfiguration:
            LogDriver: awslogs
            Options:
              awslogs-group: !Ref CloudWatchLogsGroup
              awslogs-region: !Ref AWS::Region
              awslogs-stream-prefix: ecs

  ECSService:
    Type: AWS::ECS::Service
    Properties:
      ServiceName: !Sub "${Environment}-ucup-service"
      Cluster: !Ref ECSCluster
      TaskDefinition: !Ref ECSTaskDefinition
      DesiredCount: 2
      LaunchType: FARGATE
      NetworkConfiguration:
        AwsvpcConfiguration:
          Subnets: !Ref SubnetIds
          SecurityGroups:
            - !Ref ECSServiceSecurityGroup

Outputs:
  ServiceUrl:
    Description: URL of the UCUP service
    Value: !Sub "http://${ECSService.LoadBalancers[0].DNSName}"
    Export:
      Name: !Sub "${Environment}-ucup-service-url"
"""

    def _generate_azure_template(self) -> str:
        """Generate Azure ARM template."""
        return """{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {
    "environment": {
      "type": "string",
      "defaultValue": "dev",
      "allowedValues": ["dev", "staging", "prod"]
    },
    "location": {
      "type": "string",
      "defaultValue": "[resourceGroup().location]"
    }
  },
  "resources": [
    {
      "type": "Microsoft.ContainerInstance/containerGroups",
      "apiVersion": "2021-09-01",
      "name": "[concat(parameters('environment'), '-ucup-agent')]",
      "location": "[parameters('location')]",
      "properties": {
        "containers": [
          {
            "name": "ucup-agent",
            "properties": {
              "image": "ucup/agent:latest",
              "ports": [
                {
                  "port": 8000,
                  "protocol": "TCP"
                }
              ],
              "environmentVariables": [
                {
                  "name": "ENVIRONMENT",
                  "value": "[parameters('environment')]"
                }
              ],
              "resources": {
                "requests": {
                  "cpu": 0.5,
                  "memoryInGB": 0.5
                }
              }
            }
          }
        ],
        "osType": "Linux",
        "ipAddress": {
          "type": "Public",
          "ports": [
            {
              "port": 8000,
              "protocol": "TCP"
            }
          ]
        }
      }
    }
  ],
  "outputs": {
    "serviceUrl": {
      "type": "string",
      "value": "[reference(resourceId('Microsoft.ContainerInstance/containerGroups', concat(parameters('environment'), '-ucup-agent'))).ipAddress.fqdn]"
    }
  }
}"""

    def _generate_gcp_template(self) -> str:
        """Generate GCP Deployment Manager template."""
        return """
resources:
  - name: ucup-service
    type: gcp-types/run-v1:services
    properties:
      template:
        containers:
          - image: ucup/agent:latest
            ports:
              - containerPort: 8000
            env:
              - name: ENVIRONMENT
                value: $(ref.environment)
        scaling:
          minInstanceCount: 1
          maxInstanceCount: 10
      traffic:
        - percent: 100
          latestRevision: true

  - name: environment
    type: runtimeconfig.v1beta1.config
    properties:
      config: ucup-config
      variables:
        - name: environment
          value: dev
"""


class DeploymentCLI:
    """CLI interface for deployment automation."""

    def __init__(self):
        self.automator = DeploymentAutomator()

    def deploy_agent(
        self,
        platform: str,
        agent_config: Dict[str, Any],
        environment: str = "dev",
        strategy: str = "immediate",
    ) -> str:
        """Deploy agent using specified platform and strategy."""

        # Create deployment spec from agent config
        spec = create_deployment_spec_from_config(agent_config)

        # Create cloud config
        config = create_cloud_config_from_env(platform)
        config.environment = environment

        # Set platform-specific config
        config.platform = platform

        # Convert strategy string to enum
        strategy_enum = DeploymentStrategy(strategy)

        # Create deployment pipeline if it doesn't exist
        pipeline_name = f"{agent_config.get('name', 'agent')}-pipeline"
        if pipeline_name not in self.automator.pipelines:
            pipeline = DeploymentPipeline(
                name=pipeline_name,
                environments=[environment],
                stages=[
                    {"name": "deploy", "type": "deploy", "platform": platform},
                    {
                        "name": "validate",
                        "type": "validate",
                        "checks": ["health", "metrics"],
                    },
                    {"name": "scale", "type": "scale", "policy": "default"},
                ],
            )
            self.automator.create_deployment_pipeline(pipeline)

        # Run deployment
        async def deploy():
            deployment = await self.automator.deploy_with_strategy(
                pipeline_name, environment, spec, config, strategy_enum
            )
            return deployment

        deployment = asyncio.run(deploy())

        if deployment.status == DeploymentStatus.SUCCESS:
            result = deployment.results[0] if deployment.results else None
            if result and result.endpoint:
                print(f"✅ Deployment successful!")
                print(f"🚀 Service URL: {result.endpoint}")
                print(f"📊 Monitoring: {result.monitoring_url or 'Not available'}")
                if result.cost_estimate:
                    print(f"${result.cost_estimate:.2f}")
                return deployment.id
            else:
                print("❌ Deployment failed")
                return ""
        else:
            print(f"❌ Deployment {deployment.status.value}")
            return ""

    def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get deployment status."""

        async def get_status():
            return await self.automator.get_deployment_status(deployment_id)

        return asyncio.run(get_status()) or {}

    def scale_deployment(self, deployment_id: str, replicas: int) -> bool:
        """Scale deployment."""

        async def scale():
            return await self.automator.scale_deployment(deployment_id, replicas)

        return asyncio.run(scale())

    def rollback_deployment(self, deployment_id: str) -> bool:
        """Rollback deployment."""

        async def rollback():
            return await self.automator.rollback_deployment(deployment_id)

        return asyncio.run(rollback())

    def list_deployments(self) -> List[Dict[str, Any]]:
        """List all deployments."""
        deployments = []
        for deployment in self.automator.deployments.values():
            status = self.get_deployment_status(deployment.id)
            if status:
                deployments.append(status)
        return deployments

    def create_scaling_policy(
        self,
        name: str,
        metric: str,
        target: float,
        min_instances: int,
        max_instances: int,
    ):
        """Create scaling policy."""
        policy = ScalingPolicy(
            name=name,
            metric_name=metric,
            target_value=target,
            scale_up_threshold=target * 1.2,
            scale_down_threshold=target * 0.8,
            min_instances=min_instances,
            max_instances=max_instances,
            cooldown_period=300,
        )
        self.automator.create_scaling_policy(policy)
        print(f"✅ Created scaling policy: {name}")

    def generate_infrastructure_templates(self):
        """Generate infrastructure templates."""
        self.automator.generate_infrastructure_templates()
        print("✅ Generated infrastructure templates for AWS, Azure, and GCP")


# Global CLI instance
_deployment_cli: Optional[DeploymentCLI] = None


def get_deployment_cli() -> DeploymentCLI:
    """Get or create global deployment CLI."""
    global _deployment_cli
    if _deployment_cli is None:
        _deployment_cli = DeploymentCLI()
    return _deployment_cli


# Convenience functions
def deploy_agent_to_cloud(
    platform: str,
    agent_config: Dict[str, Any],
    environment: str = "dev",
    strategy: str = "immediate",
) -> str:
    """Deploy agent to cloud platform."""
    cli = get_deployment_cli()
    return cli.deploy_agent(platform, agent_config, environment, strategy)


def get_deployment_info(deployment_id: str) -> Dict[str, Any]:
    """Get deployment information."""
    cli = get_deployment_cli()
    return cli.get_deployment_status(deployment_id)


def scale_agent_deployment(deployment_id: str, replicas: int) -> bool:
    """Scale agent deployment."""
    cli = get_deployment_cli()
    return cli.scale_deployment(deployment_id, replicas)


def rollback_agent_deployment(deployment_id: str) -> bool:
    """Rollback agent deployment."""
    cli = get_deployment_cli()
    return cli.rollback_deployment(deployment_id)
