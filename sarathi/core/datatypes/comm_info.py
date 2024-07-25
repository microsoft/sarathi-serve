from sarathi.utils import get_ip, get_random_port


class CommInfo:
    def __init__(self, driver_ip: str):
        # TODO(amey): Use a more robust method to initialize the workers.
        # In case port is already in use, this will fail.
        self.distributed_init_method = f"tcp://{driver_ip}:{get_random_port()}"
        self.engine_ip_address = get_ip()
        self.enqueue_socket_port = get_random_port()
        self.output_socket_port = get_random_port()
        self.microbatch_socket_port = get_random_port()
