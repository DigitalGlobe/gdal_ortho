import logging
import subprocess
import sys

class CommandError(Exception): pass

# Initialize logging (root level WARNING, app level INFO)
logging_format = "[%(asctime)s|%(levelname)s|%(name)s|%(lineno)d] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.WARNING, format=logging_format)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def run_cmd(args, fail_msg="Command failed", cwd=None, env=None):
    """Runs a command via subprocess and waits for it to complete.

    Args:
        args: List of args for command to run.
        fail_msg: Message to print to logger if command fails.
        cwd: Working directory for command execution.
        env: Environment variable dict for command execution.

    """

    # Convert args to strings
    string_args = [str(a) for a in args]
    logger.info("Running command: %s" % " ".join(['"%s"' % a for a in string_args]))

    # Spawn child process and wait for it to complete
    p_obj = subprocess.Popen(string_args,
                             cwd=cwd,
                             env=env)
    retval = p_obj.wait()
    if retval != 0:
        raise CommandError("%s, returned non-zero exit status %d" % \
                           (fail_msg, retval))

