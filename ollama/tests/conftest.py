import os
import time
import requests
import pytest
from datadog_checks.dev import docker_run, get_docker_hostname, get_here

# Set up test instance details
HOST = get_docker_hostname()
PORT = 11435  # Changed to 11435 to avoid conflicts
OLLAMA_URL = f"http://{HOST}:{PORT}"
INSTANCE = {"host": HOST, "port": PORT}


@pytest.fixture(scope="session")
def dd_environment():
    """
    Spins up the Ollama container using docker-compose before running tests.
    Ensures models are preloaded and a sample prompt is sent.
    """
    compose_file = os.path.join(get_here(), "docker-compose.yml")

    # Start Docker container and wait for Ollama to be ready
    with docker_run(compose_file, endpoints=[OLLAMA_URL]):
        # Give Ollama time to start up
        time.sleep(5)

        # Ensure the models are available by pulling inside the test
        try:
            print("Pulling required Ollama models...")
            requests.post(f"{OLLAMA_URL}/api/pull",
                          json={"name": "smollm:135m"}).raise_for_status()
            requests.post(f"{OLLAMA_URL}/api/pull",
                          json={"name": "all-minilm:22m"}).raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Failed to pull models: {e}")

        # Send a sample prompt to ensure "running models" data exists
        try:
            print("Sending sample prompt to initialize running models...")
            sample_payload = {
                "model": "smollm:135m",
                "prompt": "Say hello!"
            }
            requests.post(f"{OLLAMA_URL}/api/generate",
                          json=sample_payload).raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Failed to send sample prompt: {e}")

        yield INSTANCE


@pytest.fixture
def instance():
    """Ensures the same test instance is available for each test"""
    return INSTANCE.copy()
