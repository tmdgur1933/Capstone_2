import requests


class BackendClient:
    def __init__(self, backend_url: str, endpoint: str = "/api/analysis/snapshot", timeout: float = 2.0):
        self.backend_url = backend_url.rstrip("/")
        self.endpoint = endpoint
        self.timeout = timeout

    @property
    def url(self) -> str:
        if self.endpoint.startswith("/"):
            return f"{self.backend_url}{self.endpoint}"
        return f"{self.backend_url}/{self.endpoint}"

    def send_snapshot(self, snapshot: dict) -> bool:
        try:
            response = requests.post(self.url, json=snapshot, timeout=self.timeout)
            if response.status_code >= 400:
                print(f"[WARN] Backend error: {response.status_code} {response.text}")
                return False
            return True
        except requests.RequestException as e:
            print(f"[WARN] Failed to send snapshot to backend: {e}")
            return False
