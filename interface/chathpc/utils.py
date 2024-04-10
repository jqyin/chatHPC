
from typing import Optional, NamedTuple
import os, sys, subprocess, re, ast, base64, random, shutil, shlex, contextlib
from datetime import datetime, timezone
from pathlib import Path
from minio import Minio
from loguru import logger



def run_fastchat_module(module: str, args: Optional[list] = None, **kwargs):
    """
    Launch a subprocess with a fastchat module, and add the necessary env to make our models get
    loaded.
    """
    args = [str(arg) for arg in (args or [])]
    kwargs = {
        **kwargs,
        "env": {
            **os.environ,
            **kwargs.get("env", {}),
        },
    }
    module_args = [sys.executable, "-m", module, *(args or [])]
    logger.info(f"Running {module_args}")
    proc = subprocess.Popen(module_args, text=True, **kwargs)
    return proc


@contextlib.contextmanager
def set_env(**environ):
    """ Temporarily set os.environ """
    old_environ = dict(os.environ)
    os.environ.update(environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)


def run_fastchat_module_interactive(module: str, args: Optional[list] = None):
    args = [str(arg) for arg in (args or [])]
    module_args = [sys.executable, "-m", module, *(args or [])]

    logger.info(f"Running {module_args}")

    os.system(shlex.join(module_args))


def wait_for_first(*procs: subprocess.Popen):
    """ Wait for any one of multiple processes to complete. Then kill the rest. """
    pids = [proc.pid for proc in procs]
    pid = None
    while pid not in pids:
        pid, exit_code = os.wait() # Waits for any child process to finish

    # Kill all the processes if any fail
    for proc in procs:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
    
    # Wait for processes to die
    for proc in procs:
        proc.wait()


class TestMessageResponse(NamedTuple):
    loaded_models: list[str]
    response: str

def test_message(model: str, message = "Hello!"):
    """ Send a test message to the FastChat model """
    proc = run_fastchat_module("fastchat.serve.test_message",
        ["--model-name", model, "--message", message], 
        stderr = subprocess.PIPE, stdout = subprocess.PIPE,
    )
    stderr, stdout = proc.communicate()
    output = stderr # test_message uses stderr for output for some reason

    pat = r"(.*?\n)?Models: (\[.*?\])\nworker_addr: (.*?)\n(.*?)"
    match = re.fullmatch(pat, output.strip(), re.DOTALL)
    log_spam, models, worker_addr, response = match.groups()
     # Parse a python list (literal_eval is "safe" it will only evaluate literals)
    models = list(ast.literal_eval(models))
    response = response.strip()
    return TestMessageResponse(
        loaded_models=models,
        response=response,
    )


def minio_conn():
    """Create a readonly connection to Obsidian minio """
    # Default is upstream
    env_prefix = "RO_UP"
    endpoint = os.environ[f'{env_prefix}_MINIO_ENDPOINT']
    access_key = os.environ[f'{env_prefix}_MINIO_ACCESS_KEY']
    secret_key = os.environ[f'{env_prefix}_MINIO_SECRET_KEY']

    if endpoint.startswith("http://"):
        endpoint = endpoint[7:]
        secure = False
    elif endpoint.startswith("https://"):
        endpoint = endpoint[8:]
        secure = True
    else:
        raise Exception(f"invalid endpoint {endpoint}")
    mc = Minio(endpoint,
               access_key=access_key,
               secret_key=secret_key,
               secure=secure)
    return mc


def copy_atomic(src: Path, dest: Path):
    """
    Copies a file atomically by copying to a tmp file then renaming.
    """
    src, dest = Path(src), Path(dest)
    rand_str = base64.urlsafe_b64encode(random.randbytes(6)).decode()
    tmp = dest.parent / f".{dest.name}.{rand_str}.tmp"
    try:
        shutil.copy(src, tmp)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise e
    else: # No exception
        tmp.rename(dest) # Will replace file if already exists


# Allow syncing files from mounted GPFS/Lustre or from s3.
class FileSourceAdapter:
    def list_files(self, src_dir: str) -> dict[str, datetime]: return {}
    def download(self, src_dir: str, dest_dir: str, file: str): pass


class S3SourceAdapter(FileSourceAdapter):
    def __init__(self, mc: Minio):
        self.mc = mc

    def _split_url(self, s3a_url: str):
        match = re.fullmatch(f"s3a://(.*?)(/.*?)?", s3a_url)
        s3_bucket, s3_prefix = match.groups()
        s3_prefix = s3_prefix.strip("/") + "/" if s3_prefix else ""
        return s3_bucket, s3_prefix

    def list_files(self, src_dir: str) -> dict[str, datetime]:
        files: dict[str, datetime] = {}
        s3_bucket, s3_prefix = self._split_url(src_dir)

        for file in self.mc.list_objects(s3_bucket, s3_prefix, recursive=True):
            name = file.object_name.removeprefix(s3_prefix)
            files[name] = file.last_modified

        return files

    def download(self, src_dir: str, dest_dir: str, file: str):
        s3_bucket, s3_prefix = self._split_url(src_dir)
        # fget is atomic
        self.mc.fget_object(s3_bucket, s3_prefix + file, str(Path(dest_dir) / file))

# Note: This doesn't currently work as we can't seem to get both GPFS a shared NFS volume in the
# same pod. So instead I'm just using `s3a://lake` which maps to /gpfs/alpine/stf218/proj-shared/data/lake
class FileSystemSourceAdapter(FileSourceAdapter):
    def list_files(self, src_dir: str) -> dict[str, datetime]:
        path = Path(src_dir)
        files: dict[str, datetime] = {}

        for file in path.glob("**/*"):
            if file.is_file():
                name = str(file.relative_to(path))
                files[name] = datetime.fromtimestamp(file.stat().st_mtime, tz=timezone.utc)
        
        return files

    def download(self, src_dir: str, dest_dir: str, file: str):
        copy_atomic(Path(src_dir) / file, Path(dest_dir) / file)


def sync_models_from_s3(src_dir: str, dest_dir: str):
    """
    Sync an s3 or GPFS/Lustre folder to a local dir using simple mtime checks.
    Only downloads models that are needed.
    """
    logger.info(f"Syncing from {src_dir} to {dest_dir}...")
    
    if src_dir.startswith("s3a://"):
        adapter = S3SourceAdapter(minio_conn())
    else:
        adapter = FileSystemSourceAdapter()

    dest_path = Path(dest_dir).resolve()
    dest_path.mkdir(exist_ok=True)

    local_files = FileSystemSourceAdapter().list_files(dest_dir)
    remote_files = adapter.list_files(src_dir)

    # Sort local first so we remove files before downloading (save space)
    all_files = [*sorted(local_files.keys()), *sorted(remote_files.keys() - local_files.keys())]

    for file in all_files:
        in_local = file in local_files
        in_remote = file in remote_files

        if not in_local and in_remote:
            logger.info(f"Downloading {file}...")
            adapter.download(src_dir, dest_dir, file)
        elif in_local and not in_remote:
            logger.info(f"Removing {file}...")
            (dest_path / file).unlink()
        elif in_local and in_remote and remote_files[file] > local_files[file]:
            logger.info(f"Updating {file}...")
            (dest_path / file).unlink()
            adapter.download(src_dir, dest_dir, file)
        else:
            logger.info(f"{file} already up to date.")

    # Remove any empty directories left behind
    files = sorted(dest_path.glob("**"), reverse=True) # children will be before their parents
    for file in files:
        if file.is_dir() and len(list(file.iterdir())) == 0 and file != dest_path:
            logger.info(f"Removing {file}...")
            file.rmdir()

    logger.info(f"{src_dir} and {dest_dir} synced")

# We originally used s3 as we couldn't get GPFS and a mounted NFS volume in the same pod.
# That issue seems to have been resolved now so I'm just using rsync now.
def rsync(src, dest):
    # I'm just using rsync for this
    src = str(Path(src).resolve())
    dest = str(Path(dest).resolve())
    subprocess.run(["rsync", "-ra", "--delete", f"{src}/", dest], check=True)
    logger.info(f"{src} synced to {dest}")

