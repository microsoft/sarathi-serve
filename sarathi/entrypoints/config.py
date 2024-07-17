import ssl
from dataclasses import dataclass, field
from typing import Optional

from sarathi.config import BaseEndpointConfig, ReplicaConfig, SystemConfig


@dataclass
class APIServerConfig(BaseEndpointConfig):
    log_level: str = field(
        default="debug", metadata={"help": "Logging level for the API server."}
    )
    host: str = field(
        default="localhost",
        metadata={"help": "Hostname or IP address to bind the server to."},
    )
    port: int = field(
        default=8000, metadata={"help": "Port number to run the server on."}
    )
    ssl_keyfile: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the SSL key file for HTTPS connections."},
    )
    ssl_certfile: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the SSL certificate file for HTTPS connections."},
    )
    ssl_ca_certs: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the SSL CA certificate file for HTTPS connections."},
    )
    ssl_cert_reqs: int = field(
        default=int(ssl.CERT_NONE),
        metadata={
            "help": "SSL certificate requirements (CERT_NONE, CERT_OPTIONAL, or CERT_REQUIRED)."
        },
    )
    server_root_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Root path for the server (useful for reverse proxy setups)."
        },
    )

    def create_system_config(self) -> SystemConfig:
        return super().create_system_config(
            ReplicaConfig(
                output_dir=self.output_dir,
            )
        )
