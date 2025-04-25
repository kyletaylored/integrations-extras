import requests
from datadog_checks.base import AgentCheck, ConfigurationError


class OllamaCheck(AgentCheck):
    """Datadog integration for monitoring Ollama service"""

    __NAMESPACE__ = 'ollama'  # ✅ Namespace automatically applies "ollama." to all metrics

    def check(self, instance):
        """
        Runs the Ollama service check and collects model-related metrics.
        """
        # Validate configuration
        host = instance.get("host")
        port = instance.get("port")

        if not host or not port:
            raise ConfigurationError(
                "Both 'host' and 'port' must be provided in the instance configuration."
            )

        ollama_url = f"http://{host}:{port}"
        health_endpoint = f"{ollama_url}/"
        tags_endpoint = f"{ollama_url}/api/tags"
        ps_endpoint = f"{ollama_url}/api/ps"

        self.log.info(f"Starting Ollama check for {ollama_url}")

        # Health Check
        self._check_service_health(health_endpoint)

        # Collect model-related metrics
        self._collect_model_metrics(tags_endpoint, ps_endpoint)

    def _check_service_health(self, health_endpoint):
        """
        Check if Ollama is reachable and submit a service check to Datadog.
        """
        try:
            response = self.http.get(health_endpoint, timeout=5)
            response.raise_for_status()
            self.service_check("service_health", self.OK)
            self.log.info("Ollama is up and healthy.")
        except requests.exceptions.RequestException as e:
            # Only failures have messages
            self.service_check("service_health", self.CRITICAL, message=str(e))
            self.log.error("Ollama health check failed: %s", str(e))

    def _collect_model_metrics(self, tags_endpoint, ps_endpoint):
        """
        Fetch and submit model-related metrics.
        """
        # Fetch available models
        available = self._fetch_json(tags_endpoint, "models_available")
        if available:
            self._submit_available_model_metrics(available)

        # Fetch running models
        running = self._fetch_json(ps_endpoint, "models_running")
        if running:
            self._submit_running_model_metrics(running)

    def _fetch_json(self, url, metric_name):
        """
        Fetch JSON data from an Ollama API endpoint.
        """
        try:
            self.log.info(f"Making API call to: {url}")  # 🔍 Debugging log
            response = self.http.get(url, timeout=5)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            self.service_check(metric_name, self.CRITICAL, message=str(e))
            self.log.error(f"Failed to fetch data from {url}: {str(e)}")
            return None

    def _submit_available_model_metrics(self, models_data):
        """
        Submit metrics for all available models from `/api/tags`.
        """
        print(models_data)
        total_models = len(models_data.get("models", []))
        total_size_bytes = sum(model.get("size", 0)
                               for model in models_data.get("models", []))

        self.count("models.count", total_models)  # ✅ Removed manual "ollama."
        self.gauge("models.size_total_bytes", total_size_bytes)

        # Submit per-model size
        for model in models_data.get("models", []):
            model_size = model.get("size", 0)
            self.gauge("models.size_per_model", model_size)

    def _submit_running_model_metrics(self, models_data):
        """
        Submit metrics for currently running models from `/api/ps`.
        """
        print(models_data)
        running_models = models_data.get("models", [])
        running_count = len(running_models)
        total_vram_usage = sum(model.get("size_vram", 0)
                               for model in running_models)

        # Submit total running model count and VRAM usage
        self.count("models.running.count", running_count)
        self.gauge("models.running.size_vram_total", total_vram_usage)

        # Submit per-model VRAM usage
        for model in running_models:
            model_vram = model.get("size_vram", 0)
            self.gauge("models.running.size_vram_per_model", model_vram)

        # Submit running model count by family
        model_families = {}
        for model in running_models:
            family = model.get("details", {}).get("family", "unknown")
            model_families[family] = model_families.get(family, 0) + 1

        for family, count in model_families.items():
            self.count("models.running.by_family", count)
