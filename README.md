# UCUP.framework

**UCUP Framework - macOS Unified Cognitive Uncertainty Processing**

A native macOS framework that provides cognitive uncertainty processing capabilities with Core ML acceleration, Grand Central Dispatch coordination, and multimodal fusion processing.

## 🚀 Features

### Core Capabilities
- **Probabilistic Reasoning**: Core ML accelerated uncertainty processing and decision making
- **Multi-Agent Coordination**: GCD-based hierarchical, swarm, and adaptive coordination strategies
- **Multimodal Processing**: Vision and AVFoundation integration for comprehensive sensory fusion
- **Real-time Performance**: Native macOS optimizations for low-latency processing

### Comprehensive Testing Framework
- **Agent Testing**: Performance benchmarking, stress testing, and validation for probabilistic agents
- **Performance Validation**: Threshold validation, implementation comparison, and statistical analysis
- **Scenario Generation**: Realistic test scenarios for high-load, failure injection, resource constraints, and network issues

### Full Observability Stack
- **System Monitoring**: Real-time health checks, alerts, and system metrics collection
- **Metrics Collection**: Comprehensive metrics aggregation with percentile analysis and time-series data
- **Decision Visualization**: Interactive HTML visualizations for probabilistic decisions, coordination flows, and performance metrics

## 📦 Installation

### Building from Source

```bash
# Clone the repository
git clone <repository-url>
cd UCUP.framework

# Build the framework
python build.py

# Run tests
python test_ucup.py
```

### System Requirements
- **macOS 10.15+** (Catalina or later)
- **Python 3.8+**
- **Xcode** (optional - framework gracefully falls back to development mode)

## 🏗️ Architecture

```
UCUP.framework/
├── Headers/                 # Objective-C headers
├── Resources/              # Framework resources and Info.plist
├── Versions/A/             # Current framework version
│   ├── PythonModules/ucup/ # Python framework modules
│   │   ├── core/          # Core framework components
│   │   ├── probabilistic/ # ML reasoning engine
│   │   ├── coordination/  # Multi-agent coordination
│   │   ├── multimodal/    # Sensory fusion processing
│   │   ├── testing/       # Comprehensive testing suite
│   │   └── observability/ # Monitoring and visualization
│   └── UCUP               # Objective-C binary
└── UCUP -> Versions/Current/UCUP  # Symlink to current binary
```

## 📚 API Overview

### Quick Start

```python
import ucup

# Initialize the framework
manager = ucup.initialize_ucup()

# Create a probabilistic agent
agent = ucup.create_probabilistic_agent()

# Create a multimodal processor
processor = ucup.create_multimodal_processor()

# Create a coordinator
coordinator = ucup.create_coordinator("hierarchical")
```

### Core Components

#### Probabilistic Processing
```python
from ucup import ProbabilisticAgent, ProbabilisticEngine

# Create and configure agent
agent = ProbabilisticAgent(manager)
result = agent.predict({"input_data": 0.75})
# Returns: {"probability": 0.82, "confidence_interval": [0.78, 0.86]}
```

#### Multimodal Fusion
```python
from ucup import MultimodalFusionEngine

# Process multimodal inputs
engine = MultimodalFusionEngine(manager)
result = engine.fuse_multimodal_data([
    {"type": "vision", "data": image_data},
    {"type": "audio", "data": audio_data}
])
```

#### Agent Coordination
```python
from ucup import HierarchicalCoordinator, SwarmCoordinator

# Hierarchical coordination
hierarchical = HierarchicalCoordinator(manager)
result = hierarchical.execute_task(task_definition)

# Swarm coordination
swarm = SwarmCoordinator(manager)
result = swarm.coordinate_agents(agent_list)
```

### Testing Framework

```python
from ucup.testing import AgentTester, PerformanceValidator, ScenarioGenerator

# Test agent performance
tester = AgentTester(manager)
results = tester.test_probabilistic_agent(agent, test_cases)

# Validate performance thresholds
validator = PerformanceValidator(manager)
validation = validator.validate_performance_thresholds(
    operation=my_operation,
    operation_name="my_process",
    thresholds={"average_execution_time": 0.1}
)

# Generate test scenarios
generator = ScenarioGenerator(manager)
scenario = generator.generate_scenario("high_load", duration=300, intensity=0.8)
```

### Observability

```python
from ucup.observability import SystemMonitor, MetricsCollector, DecisionVisualizer

# System monitoring
monitor = SystemMonitor(manager)
monitor.start_monitoring(interval=5.0)
health = monitor.get_health_status()

# Metrics collection
collector = MetricsCollector(manager)
collector.start_collection()
collector.record_metric("custom_metric", 42.0, tags={"component": "test"})

# Decision visualization
visualizer = DecisionVisualizer(manager)
filepath = visualizer.visualize_probabilistic_decision(decision_data)
# Opens interactive HTML visualization
```

## 🧪 Testing

The framework includes comprehensive testing capabilities:

```bash
# Run the full test suite
python test_ucup.py

# Expected output:
# 🧪 UCUP Framework Test Suite
# ========================================
# Testing basic imports... ✅
# Testing configuration system... ✅
# Testing probabilistic engine... ⚠️ (expected in development mode)
# Testing Objective-C bridge... ⚠️ (expected in development mode)
# Testing multimodal processing... ⚠️ (expected in development mode)
#
# Test Results: 3/5 tests passed
```

**Note**: Some tests show warnings about the Objective-C bridge in development mode. This is expected when Xcode is not available for full native compilation.

## 📊 Performance

- **Core ML Acceleration**: Hardware-accelerated probabilistic computations
- **GCD Optimization**: Native macOS threading and concurrency
- **Memory Efficient**: Optimized for real-time multimodal processing
- **Scalable**: Supports multiple coordination strategies for various scales

## 🔧 Configuration

```python
from ucup import UCUPConfig

# Load custom configuration
config = UCUPConfig.load("path/to/config.json")

# Or use defaults
config = UCUPConfig.default()

# Configure components
config.set("core_ml.enabled", True)
config.set("gcd.enable_threading", True)
config.set("probabilistic.model_path", "path/to/model.mlmodel")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Built on native macOS frameworks (Core ML, Vision, AVFoundation)
- Inspired by cognitive uncertainty processing research
- Contributors: UCUP Framework Community

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the framework documentation in `Versions/A/PythonModules/ucup/`
- Review test examples in `test_ucup.py`

---

**Version**: 1.0.0
**Platform**: macOS 10.15+
**Python**: 3.8+
**License**: Apache 2.0
