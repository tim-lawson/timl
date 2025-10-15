import logging
import os
import shutil
from multiprocessing import Process, Queue
from typing import Generic, Protocol, TypeVar

import torch
from huggingface_hub import HfApi
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from transformers.modeling_utils import PreTrainedModel
from transformers.utils.logging import disable_progress_bar, enable_progress_bar

from timl.distributed import is_main_process

logger = logging.getLogger(__name__)


class UploadCheckpointArgs(Protocol):
    output_dir: str
    push_to_hub: bool
    hub_model_id: str | None
    hub_token: str | bool | None
    hub_private: bool | None
    debug: bool


T_UploadCheckpointArgs = TypeVar("T_UploadCheckpointArgs", bound=UploadCheckpointArgs)


class UploadCheckpointMixin(Generic[T_UploadCheckpointArgs]):
    args: T_UploadCheckpointArgs

    model: PreTrainedModel
    optimizer: Optimizer | None = None
    lr_scheduler: LRScheduler | None = None

    def save_checkpoint(self, step: int, revision: str | None = None) -> None:
        if is_main_process():
            revision = revision or f"step{step}"
            save_dir = os.path.join(self.args.output_dir, revision)
            os.makedirs(save_dir, exist_ok=True)

            disable_progress_bar()
            self.model.save_pretrained(save_dir)
            logger.info(f"Saved pretrained model to {save_dir}.")
            enable_progress_bar()

            if self.optimizer is None:
                raise RuntimeError("No optimizer found, cannot save state.")
            if self.lr_scheduler is None:
                raise RuntimeError("No LR scheduler found, cannot save state.")

            save_obj = {
                "step": step,
                "args": self.args.__dict__,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
            }
            save_path = os.path.join(save_dir, "state.pt")
            torch.save(save_obj, save_path)
            logger.info(f"Saved checkpoint to {save_path}.")

            if self.args.debug:
                shutil.rmtree(save_dir)
                logger.info(f"Removed checkpoint folder {save_dir}.")
            else:
                self.upload_checkpoint(save_dir, revision)

    def create_upload_queue(self) -> None:
        self.repo_id = self.args.hub_model_id or os.path.basename(self.args.output_dir)
        self.upload_process, self.upload_queue = None, None

        if is_main_process() and self.args.push_to_hub:
            self.upload_queue = Queue()
            upload_worker_args = (self.upload_queue, self.repo_id, self.args.hub_token)
            self.upload_process = Process(
                target=self.upload_worker, args=upload_worker_args, daemon=True
            )
            self.upload_process.start()

            api = HfApi(token=self.args.hub_token)
            if not self.args.debug and not api.repo_exists(
                self.repo_id, repo_type="model"
            ):
                api.create_repo(
                    self.repo_id, repo_type="model", private=self.args.hub_private
                )
                logger.info(f"Created Hugging Face repository {self.repo_id}.")

    @staticmethod
    def upload_worker(queue: Queue, repo_id: str, token: str) -> None:
        while True:
            item = queue.get()
            if item is None:  # Sentinel value
                break
            folder_path, revision = item
            try:
                disable_progress_bar()
                api = HfApi(token=token)
                api.create_branch(
                    repo_id, branch=revision, repo_type="model", exist_ok=True
                )
                api.upload_folder(
                    repo_id=repo_id,
                    folder_path=folder_path,
                    repo_type="model",
                    revision=revision,
                )
                logger.info(f"Uploaded folder {folder_path}.")
            except Exception as exception:
                logger.error(f"Error uploading folder: {exception}")
            finally:
                shutil.rmtree(folder_path)
                logger.info(f"Removed temporary folder {folder_path}.")
                enable_progress_bar()

    def upload_checkpoint(self, save_dir: str, revision: str) -> None:
        if is_main_process() and self.args.push_to_hub:
            folder_path = save_dir + "-temp"
            shutil.copytree(save_dir, folder_path, dirs_exist_ok=True)

            if self.upload_queue is not None:
                self.upload_queue.put((folder_path, revision))
                logger.info(f"Queued folder {folder_path} for upload.")
            else:
                shutil.rmtree(folder_path)
                logger.error("No `upload_queue` found, cannot upload.")
                logger.info(f"Removed temporary folder {folder_path}.")

    def destroy_upload_queue(self) -> None:
        if self.upload_process is not None:
            if self.upload_queue is not None:
                self.upload_queue.put(None)  # Sentinel value
            logger.info("Waiting for uploads to complete...")
            self.upload_process.join()
            logger.info("Uploads completed.")
