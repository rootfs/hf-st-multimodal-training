#!/usr/bin/env python3

import argparse
import ctypes
import os
import subprocess
import sys
from ctypes import c_char_p, c_int, c_uint32


IN_CLOSE_WRITE = 0x00000008
IN_CREATE = 0x00000100
IN_MOVED_TO = 0x00000080
IN_ONLYDIR = 0x01000000
IN_NONBLOCK = 0x00000800
WATCH_MASK = IN_CREATE | IN_MOVED_TO | IN_CLOSE_WRITE


class InotifyWatcher:
    def __init__(self, path: str) -> None:
        self._libc = ctypes.CDLL("libc.so.6", use_errno=True)
        self._libc.inotify_init1.argtypes = [c_int]
        self._libc.inotify_init1.restype = c_int
        self._libc.inotify_add_watch.argtypes = [c_int, c_char_p, c_uint32]
        self._libc.inotify_add_watch.restype = c_int
        self.fd = self._libc.inotify_init1(IN_NONBLOCK)
        if self.fd < 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err))
        watch_result = self._libc.inotify_add_watch(self.fd, path.encode("utf-8"), WATCH_MASK | IN_ONLYDIR)
        if watch_result < 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err), path)

    def fileno(self) -> int:
        return self.fd

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


def final_export_ready(final_dir: str) -> bool:
    return os.path.exists(os.path.join(final_dir, "model.pt")) and os.path.exists(os.path.join(final_dir, "config.json"))


def build_upload_command(args: argparse.Namespace) -> str:
    pieces = [
        "python /workspace/app/scripts/upload_tri_encoder_to_hf.py",
        f"--final-dir {shell_quote(args.final_dir)}",
        f"--repo-id {shell_quote(args.repo_id)}",
        f"--token-name {shell_quote(args.token_name)}",
    ]
    if args.max_samples is not None:
        pieces.append(f"--max-samples {args.max_samples}")
    if args.min_eval_top1 is not None:
        pieces.append(f"--min-eval-top1 {args.min_eval_top1}")
    if args.max_eval_loss is not None:
        pieces.append(f"--max-eval-loss {args.max_eval_loss}")
    return " ".join(pieces)


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def run_upload(args: argparse.Namespace) -> int:
    inner = build_upload_command(args)
    command = [
        "docker",
        "compose",
        "run",
        "--rm",
        "trainer",
        "bash",
        "-lc",
        inner,
    ]
    print(f"[watcher] final export detected, launching: {inner}", flush=True)
    result = subprocess.run(command, cwd=args.project_dir)
    return result.returncode


def wait_for_final(args: argparse.Namespace) -> int:
    if final_export_ready(args.final_dir):
        print("[watcher] final export already present", flush=True)
        return run_upload(args)

    parent_dir = os.path.dirname(args.final_dir.rstrip(os.sep))
    os.makedirs(parent_dir, exist_ok=True)
    watcher = InotifyWatcher(parent_dir)
    try:
        print(f"[watcher] waiting for final export under {args.final_dir}", flush=True)
        while True:
            if final_export_ready(args.final_dir):
                return run_upload(args)
            os.read(watcher.fileno(), 4096)
    except BlockingIOError:
        import select

        poller = select.poll()
        poller.register(watcher.fileno(), select.POLLIN)
        while True:
            if final_export_ready(args.final_dir):
                return run_upload(args)
            poller.poll()
            try:
                os.read(watcher.fileno(), 4096)
            except BlockingIOError:
                pass
    finally:
        watcher.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wait for the tri-encoder final export, then run gated eval and Hugging Face upload."
    )
    parser.add_argument(
        "--project-dir",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        help="Path to the hf-st-multimodal-server project containing docker-compose.yml",
    )
    parser.add_argument(
        "--final-dir",
        default="/scratch/hf_st_mm_outputs/server_datacenter_8gpu_tri_encoder/final",
        help="Final export directory to watch for model.pt and config.json",
    )
    parser.add_argument(
        "--repo-id",
        default="llm-semantic-router/multi-modal-embed-large",
        help="Destination Hugging Face model repository",
    )
    parser.add_argument(
        "--token-name",
        default="model training",
        help="Named entry in ~/.cache/huggingface/stored_tokens used by the uploader",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optionally evaluate only the first N validation samples before upload",
    )
    parser.add_argument(
        "--min-eval-top1",
        type=float,
        default=0.85,
        help="Minimum eval_top1 required before upload",
    )
    parser.add_argument(
        "--max-eval-loss",
        type=float,
        default=0.45,
        help="Maximum eval_loss allowed before upload",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exit_code = wait_for_final(args)
    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()