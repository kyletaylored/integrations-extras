from unittest.mock import patch, Mock
import pytest
import requests

from datadog_checks.ollama import OllamaCheck
from datadog_checks.base import ConfigurationError, AgentCheck


# -------------------------
# UNIT TESTS
# -------------------------

@pytest.mark.unit
def test_config():
    """Test configuration validation inside check()"""

    check = OllamaCheck('ollama', {}, [])

    # Missing both host and port should raise ConfigurationError
    with pytest.raises(ConfigurationError):
        check.check({})

    # Missing port should raise ConfigurationError
    with pytest.raises(ConfigurationError):
        check.check({'host': 'localhost'})

    # Missing host should raise ConfigurationError
    with pytest.raises(ConfigurationError):
        check.check({'port': 11434})

    # Valid configuration should not raise an error
    check.check({'host': 'localhost', 'port': 11434})


@pytest.mark.unit
@patch("requests.get")  # ✅ Patch requests.get globally
def test_health_check_ok(mock_get):
    """Test service health check when Ollama is available"""

    instance = {'host': 'localhost', 'port': 11434}
    check = OllamaCheck('ollama', {}, [])

    # Configure the mock response for health check
    mock_response = Mock()
    mock_response.status_code = 200
    mock_get.return_value = mock_response  # ✅ Mocking the GET request

    with patch.object(check, "service_check") as mock_service_check, \
            patch.object(check, "_collect_model_metrics", return_value=None):  # ✅ Prevent real API calls for metrics

        check.check(instance)

        # Ensure service check reports OK when Ollama is available
        mock_service_check.assert_called_with("service_health", AgentCheck.OK)


@pytest.mark.unit
@patch("requests.get")  # ✅ Patch requests.get globally
def test_health_check_failure(mock_get):
    """Test service health check failure when Ollama is unreachable"""

    instance = {'host': 'localhost', 'port': 11434}
    check = OllamaCheck('ollama', {}, [])

    # Simulate a connection failure
    mock_get.side_effect = requests.exceptions.ConnectionError(
        "Failed to connect")

    with patch.object(check, "service_check") as mock_service_check, \
            patch.object(check, "_collect_model_metrics", return_value=None):  # ✅ Prevent real API calls for metrics

        check.check(instance)

        # Ensure service check reports CRITICAL when Ollama is unavailable
        mock_service_check.assert_any_call(
            "service_health", AgentCheck.CRITICAL, message="Failed to connect")


@pytest.mark.unit
def test_model_metrics_submission():
    """Test if Ollama model-related metrics are correctly fetched and submitted"""

    instance = {'host': 'localhost', 'port': 11434}
    check = OllamaCheck('ollama', {}, [])

    # Mock API responses
    mock_ps_response = {
        "models": [
            {
                "name": "model-a",
                "model": "model-a",
                "size": 996335360,
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": "llama",
                    "families": ["llama"],
                    "parameter_size": "134.52M",
                    "quantization_level": "Q4_0"
                },
                "size_vram": 996335360
            },
            {
                "name": "model-b",
                "model": "model-b",
                "size": 9418521088,
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": "llama",
                    "families": None,
                    "parameter_size": "7B",
                    "quantization_level": "Q4_0"
                },
                "size_vram": 9418521088
            }
        ]
    }

    mock_tags_response = {
        "models": [
            {
                "name": "model-c",
                "model": "model-c",
                "size": 45960996,
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": "bert",
                    "families": ["bert"],
                    "parameter_size": "23M",
                    "quantization_level": "F16"
                }
            },
            {
                "name": "model-a",
                "model": "model-a",
                "size": 91739413,
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": "llama",
                    "families": ["llama"],
                    "parameter_size": "134.52M",
                    "quantization_level": "Q4_0"
                }
            },
            {
                "name": "model-b",
                "model": "model-b",
                "size": 3825910662,
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": "llama",
                    "families": None,
                    "parameter_size": "7B",
                    "quantization_level": "Q4_0"
                }
            }
        ]
    }

    with patch.object(check, "count") as mock_count, \
            patch.object(check, "gauge") as mock_gauge, \
            patch.object(check, "_fetch_json") as mock_fetch_json:

        # Ensure API calls return **separate** objects per request
        mock_fetch_json.side_effect = lambda url, metric: mock_tags_response if "tags" in url else mock_ps_response

        # Run the check
        check.check(instance)

        # Validate model count and total size.
        mock_count.assert_any_call(
            "models.count", len(mock_tags_response["models"]))
        mock_gauge.assert_any_call("models.size_total_bytes", sum(
            m["size"] for m in mock_tags_response["models"]))

        # Validate individual model size
        for model in mock_tags_response["models"]:
            mock_gauge.assert_any_call(
                "models.size_per_model", model["size"])

        # Validate running model count and VRAM usage
        mock_count.assert_any_call(
            "models.running.count", len(mock_ps_response["models"]))

        # Compute actual expected total VRAM dynamically
        expected_total_vram = sum(m["size_vram"]
                                  for m in mock_ps_response["models"])
        print(f"Expected VRAM total: {expected_total_vram}")
        mock_gauge.assert_any_call(
            "models.running.size_vram_total", expected_total_vram)

        # Validate individual running model VRAM usage
        for model in mock_ps_response["models"]:
            mock_gauge.assert_any_call(
                "models.running.size_vram_per_model", model["size_vram"])


# -------------------------
# INTEGRATION TESTS
# -------------------------


@pytest.mark.integration
@pytest.mark.usefixtures("dd_environment")
def test_service_check(aggregator, instance):
    """
    Run the Ollama check against a real containerized Ollama service.
    """

    check = OllamaCheck("ollama", {}, [instance])

    # Run the check
    check.check(instance)

    # Verify that Ollama service health is reported correctly
    aggregator.assert_service_check("ollama.service_health", AgentCheck.OK)


@pytest.mark.integration
@pytest.mark.usefixtures("dd_environment")
def test_metrics_collection(aggregator, instance):
    """
    Test that Ollama reports expected model metrics.
    """
    check = OllamaCheck("ollama", {}, [instance])

    # Run the check
    check.check(instance)

    # Validate that at least some metrics are collected
    aggregator.assert_metric("ollama.models.count")
    aggregator.assert_metric("ollama.models.size_total_bytes")
    aggregator.assert_metric("ollama.models.running.count")
    aggregator.assert_metric("ollama.models.running.size_vram_total")
