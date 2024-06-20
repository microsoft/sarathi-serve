import ssl
from dataclasses import dataclass

from sarathi.config import BaseEndpointConfig, ReplicaConfig, SystemConfig


@dataclass
class APIServerConfig(BaseEndpointConfig):
    log_level: str = "debug"
    host: str = "localhost"
    port: int = 8000
    ssl_keyfile: str = None
    ssl_certfile: str = None
    ssl_ca_certs: str = None
    ssl_cert_reqs: int = int(ssl.CERT_NONE)
    server_root_path: str = None

    def create_system_config(self) -> SystemConfig:
        return super().create_system_config(
            ReplicaConfig(
                output_dir=self.output_dir,
            )
        )
