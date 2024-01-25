import os
import traceback
import multiprocessing as mp
from vllm.utils import set_cuda_visible_devices
from vllm.logger import init_logger

logger = init_logger(__name__)


class LocalWorkerVllm(mp.Process):
    """Local process wrapper for vllm.worker.Worker 
    for handling single-node multi-GPU tensor parallel setup."""

    def __init__(self):
        super().__init__(daemon=True)
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.worker = None

    def run(self):
        # Accept tasks from the engine in task_queue
        # and return task output in result_queue
        logger.info(
            f"Worker {mp.current_process().name} pid {os.getpid()} ready; awaiting tasks")
        try:
            for items in iter(self.task_queue.get, "TERMINATE"):
                method = items[0]
                args = items[1:]
                executor = getattr(self, method)
                executor(*args)
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(
                f"Exception in worker {mp.current_process().name}: {e}, {tb}")

    def set_cuda_visible_devices(self, device_id):
        set_cuda_visible_devices(device_id)

    def init_worker(self, *args):
        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        from vllm.worker.worker import Worker
        self.worker = Worker(*args)

    def execute_method(self, method, method_args):
        args, kwargs = method_args
        executor = getattr(self.worker, method)
        self.result_queue.put(executor(*args, **kwargs))
