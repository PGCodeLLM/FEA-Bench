from __future__ import annotations

import docker
from docker.types import Mount
import json
import os
import platform
import traceback
import threading
import shlex

if platform.system() == 'Linux':
    import resource

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path, PurePosixPath

from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    DOCKER_PATCH,
    DOCKER_USER,
    DOCKER_WORKDIR,
    INSTANCE_IMAGE_BUILD_DIR,
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    LOG_REPORT,
    LOG_INSTANCE,
    LOG_TEST_OUTPUT,
    RUN_EVALUATION_LOG_DIR,
    UTF8,
)
from swebench.harness.docker_utils import (
    clean_images,
    cleanup_container,
    copy_to_container,
    exec_run_with_timeout,
    list_images,
    remove_image,
    should_remove,
)
from swebench.harness.docker_build import (
    BuildImageError,
    build_container,
    build_env_images,
    close_logger,
    setup_logger,
)
from swebench.harness.grading import get_eval_report
from swebench.harness.reporting import make_run_report
from swebench.harness.modal_eval import (
    run_instances_modal,
    validate_modal_credentials,
)
from swebench.harness.test_spec.test_spec import make_test_spec, TestSpec
from swebench.harness.utils import (
    EvaluationError,
    load_swebench_dataset,
    get_predictions_from_file,
    run_threadpool,
    str2bool,
)

GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
]

# Template script for running agents
AGENT_SCRIPT_TEMPLATE = """#!/bin/bash
set -uxo pipefail
source ~/.bashrc
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff
source /opt/miniconda3/bin/activate
conda activate testbed
pip install -e .
{agent_command}
"""


class Agent:
    """Base class for agentic tools."""
    
    def __init__(self, instance: dict, input_text_key: str = 'problem_statement', api_key: str = None, base_url: str = None, model_name_or_path: str = None):
        """
        Initialize the agent with instance data.
        
        Args:
            instance (dict): Instance data containing problem_statement, repo, etc.
            input_text_key (str): Key in instance dict that contains the problem statement
            api_key (str): API key for the agent (required)
            base_url (str): Base URL for the agent API (required)
            model_name_or_path (str): Model name or path to use
            
        Raises:
            ValueError: If api_key or base_url is not provided
        """
        if api_key is None:
            raise ValueError("API key is required. Please set OPENAI_API_KEY environment variable.")
        if base_url is None:
            raise ValueError("Base URL is required. Please set OPENAI_BASE_URL environment variable.")
            
        self.instance = instance
        self.problem_statement = instance.get(input_text_key, '')
        self.api_key = api_key
        self.base_url = base_url
        self.model = model_name_or_path
    
    def install(self, container, logger, workdir: str, user: str, use_cache: bool = True) -> bool:
        """
        Install the agent and its dependencies in the container.
        
        Args:
            container: Docker container object
            logger: Logger instance
            workdir: Working directory in the container
            user: User to run commands as
            use_cache: Whether to use cached installation (applicable to some agents)
            
        Returns:
            bool: True if installation was successful, False otherwise
        """
        raise NotImplementedError("Subclasses must implement install()")
    
    def get_cli_command(self) -> str:
        """
        Build and return just the agent CLI command (without shell wrapper).
        
        Returns:
            str: The CLI command to execute the agent
        """
        raise NotImplementedError("Subclasses must implement get_cli_command()")


class IFlowAgent(Agent):
    """iFlow CLI agent implementation."""
    
    def install(self, container, logger, workdir: str, user: str, use_cache: bool = True, cached_archive_path: str = "iflow-cli-npm.tar.gz") -> bool:
        """Install iflow-cli agent."""
        logger.info("Installing iflow-cli agent...")
        
        if use_cache:
            # Alternative installation: extract pre-packaged tar.gz
            logger.info("Using tar.gz-based installation...")
            
            # Copy iflow-cli-npm.tar.gz to /root in container using put_archive directly
            logger.info(f"Copying {cached_archive_path} to container...")
            tar_file = Path(cached_archive_path)
            if not tar_file.exists():
                logger.error(f"{cached_archive_path} not found in current directory")
                return False
            
            # Read the tar.gz file and use put_archive directly
            with open(tar_file, "rb") as f:
                data = f.read()
            
            # put_archive extracts the tar automatically to the specified path
            container.put_archive("/root", data)
            logger.info(f"{cached_archive_path} copied and extracted successfully")
            
            # Append paths to /root/.bashrc
            logger.info("Updating /root/.bashrc with environment variables...")
            bashrc_append = '''\nexport NVM_DIR="/root/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"
export PATH="$HOME/.npm-global/bin:/root/.nvm/versions/node/v22.22.0/bin:$PATH"
'''
            
            append_result = container.exec_run(
                f'bash -c "echo {shlex.quote(bashrc_append)} >> /root/.bashrc"',
                workdir="/root",
                user=user
            )
            
            if append_result.exit_code != 0:
                logger.warning(f"Failed to update .bashrc: {append_result.output.decode('utf-8')}")
                return False
            
            logger.info("/root/.bashrc updated successfully")
        else:
            # Installing iflow-cli using npm - removing running cli at the end of install script to avoid interactive prompt
            # and adding npm timeout configuration
            logger.info("Using npm-based installation...")
            install_result = container.exec_run(
                'bash -c "curl -fsSL https://cloud.iflow.cn/iflow-cli/install.sh | sed \'/^[[:space:]]*iflow[[:space:]]*$/d\' | sed \'/install_iFlow_cli() {/a\\    log_info Configuring npm timeout settings...\\n    npm config set fetch-timeout 600000\\n    npm config set fetch-retry-mintimeout 20000\\n    npm config set fetch-retry-maxtimeout 120000\\n    npm config set fetch-retries 5\' | bash"',
                workdir=workdir,
                user=user
            )

            logger.debug("iflow-cli installation output:")
            logger.debug(install_result.output.decode('utf-8'))
            
            if install_result.exit_code != 0:
                logger.warning(f"Failed to install iflow-cli: {install_result.output.decode('utf-8')}")
                return False
        
        logger.info("iflow-cli installed successfully")
        
        # Configure iflow-cli settings
        logger.info("Configuring iflow-cli settings...")
        config_command = f"""python3 -c "
import json
import os

config_path = '/root/.iflow/settings.json'
os.makedirs(os.path.dirname(config_path), exist_ok=True)

# Read existing config or start with empty dict
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
else:
    config = {{}}

# Update configuration
config['selectedAuthType'] = 'openai-compatible'
config['apiKey'] = '{self.api_key}'
config['baseUrl'] = '{self.base_url}'
config['modelName'] = '{self.model}'

# Write back to file
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print('Configuration updated successfully')
"
"""
        config_result = container.exec_run(
            f'bash -c {shlex.quote(config_command)}',
            workdir=workdir,
            user=user
        )
        
        logger.debug("Configuration output:")
        logger.debug(config_result.output.decode('utf-8'))
        
        if config_result.exit_code != 0:
            logger.warning(f"Failed to configure iflow-cli: {config_result.output.decode('utf-8')}")
            return False
        
        logger.info("iflow-cli configured successfully")
        return True
    
    def get_cli_command(self) -> str:
        """Build iflow-cli execution command."""
        # Escape single quotes in problem statement for bash
        escaped_problem = shlex.quote(self.problem_statement)
        return f'iflow -p {escaped_problem} --output-file /logs/exec_info.json --telemetry --telemetry-outfile /logs/telemetry_traces.txt --telemetry-log-prompts'


class SWEAgent(Agent):
    """SWE-agent implementation."""
    
    def install(self, container, logger, workdir: str, user: str) -> bool:
        """Install SWE-agent."""
        logger.info("Installing SWE-agent...")
        
        install_result = container.exec_run(
            "pip install sweagent",
            workdir=workdir,
            user=user
        )
        
        if install_result.exit_code != 0:
            logger.warning(f"Failed to install sweagent: {install_result.output.decode('utf-8')}")
            return False
        
        logger.info("SWE-agent installed successfully")
        return True
    
    def get_cli_command(self) -> str:
        """Build SWE-agent execution command."""
        escaped_problem = shlex.quote(self.problem_statement)
        return f"sweagent --problem {escaped_problem} --output /output.patch"


class OpenHandsAgent(Agent):
    """OpenHands agent implementation."""
    
    def install(self, container, logger, workdir: str, user: str) -> bool:
        """Install OpenHands agent."""
        logger.info("Installing OpenHands agent...")
        
        install_result = container.exec_run(
            "pip install openhands",
            workdir=workdir,
            user=user
        )
        
        if install_result.exit_code != 0:
            logger.warning(f"Failed to install openhands: {install_result.output.decode('utf-8')}")
            return False
        
        logger.info("OpenHands installed successfully")
        return True
    
    def get_cli_command(self) -> str:
        """Build OpenHands execution command."""
        escaped_problem = shlex.quote(self.problem_statement)
        return f"openhands --problem {escaped_problem} --output /output.patch"


class ClaudeCodeAgent(Agent):
    """Claude Code agent implementation."""
    
    def install(self, container, logger, workdir: str, user: str, use_cache: bool = True) -> bool:
        """Install Claude Code agent."""
        logger.info("Installing Claude Code agent...")
        
        # Install Claude Code using the official installation script
        logger.info("Using curl-based installation...")
        install_result = container.exec_run(
            'bash -c "curl -fsSL https://claude.ai/install.sh | bash"',
            workdir=workdir,
            user=user
        )
        
        logger.debug("Claude Code installation output:")
        logger.debug(install_result.output.decode('utf-8'))
        
        if install_result.exit_code != 0:
            logger.warning(f"Failed to install Claude Code: {install_result.output.decode('utf-8')}")
            return False
        
        # Add Claude Code to PATH in .bashrc
        logger.info("Updating /root/.bashrc with PATH...")
        bashrc_append = '\nexport PATH="$HOME/.local/bin:$PATH"\n'
        
        append_result = container.exec_run(
            f'bash -c "echo {shlex.quote(bashrc_append)} >> /root/.bashrc"',
            workdir="/root",
            user=user
        )
        
        if append_result.exit_code != 0:
            logger.warning(f"Failed to update .bashrc: {append_result.output.decode('utf-8')}")
            return False
        
        logger.info("/root/.bashrc updated successfully")

        # Create symlink from ~/.claude/projects/ to /logs
        logger.info("Creating symlink from ~/.claude/projects/ to /logs...")
        symlink_result = container.exec_run(
            'bash -c "mkdir -p ~/.claude && ln -sf /logs ~/.claude/projects"',
            workdir="/root",
            user=user
        )

        if symlink_result.exit_code != 0:
            logger.warning(f"Failed to create symlink: {symlink_result.output.decode('utf-8')}")
            return False

        logger.info("Symlink created successfully")

        logger.info("Claude Code installed successfully")
        return True
    
    def get_cli_command(self) -> str:
        """Build Claude Code execution command with environment variables."""
        escaped_problem = shlex.quote(self.problem_statement)
        # Remove trailing /v1 in base_url if present, as claude CLI expects base URL without version path
        base_url = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        # Pass authentication via environment variables
        return f'ANTHROPIC_AUTH_TOKEN="{self.api_key}" ANTHROPIC_BASE_URL="{base_url}" ANTHROPIC_MODEL="{self.model}" ANTHROPIC_API_KEY="" IS_SANDBOX=1 claude --dangerously-skip-permissions -p {escaped_problem}'


# Agent registry for easy lookup
AGENT_REGISTRY = {
    "iflow-cli": IFlowAgent,
    "sweagent": SWEAgent,
    "openhands": OpenHandsAgent,
    "claude-code": ClaudeCodeAgent,
}


def get_agent(agent_name: str, instance: dict, input_text_key: str = 'problem_statement', model_name_or_path: str = None) -> Agent:
    """
    Factory function to get the appropriate agent instance.
    
    Args:
        agent_name (str): Name of the agent
        instance (dict): Instance data
        input_text_key (str): Key in instance dict that contains the problem statement
        model_name_or_path (str): Model name or path to use
        
    Returns:
        Agent: An instance of the appropriate agent class
        
    Raises:
        ValueError: If agent_name is not recognized or required environment variables are missing
    """
    # Read API credentials from environment
    api_key = os.environ.get('OPENAI_API_KEY')
    base_url = os.environ.get('OPENAI_BASE_URL')
    
    agent_class = AGENT_REGISTRY.get(agent_name)
    if agent_class is None:
        raise ValueError(f"Unknown agent name: {agent_name}. Available agents: {list(AGENT_REGISTRY.keys())}")
    return agent_class(instance, input_text_key, api_key=api_key, base_url=base_url, model_name_or_path=model_name_or_path)


def run_instance(
        test_spec: TestSpec,
        instance: dict,
        agent: Agent,
        rm_image: bool,
        force_rebuild: bool,
        client: docker.DockerClient,
        run_id: str,
        model_nickname: str,
        timeout: int | None = None,
        output_file: str = None,
        file_lock: threading.Lock = None,
        logs_base_path: str = None,
    ):
    """
    Run a single instance with the agent.

    Args:
        test_spec (TestSpec): TestSpec instance
        instance (dict): Instance data with repo, base_commit, problem_statement, etc.
        agent (Agent): Agent instance to run
        rm_image (bool): Whether to remove the image after running
        force_rebuild (bool): Whether to force rebuild the image
        client (docker.DockerClient): Docker client
        run_id (str): Run ID
        model_nickname (str): Model nickname for log dir naming
        timeout (int): Timeout for running agent
        output_file (str): Output file path to write results
        file_lock (threading.Lock): Lock for thread-safe file writing
        logs_base_path (str): Base path to prepend to log directory
    """
    # Set up logging directory
    instance_id = test_spec.instance_id
    agent_name = agent.__class__.__name__
    base_log_dir = Path(logs_base_path) / RUN_EVALUATION_LOG_DIR if logs_base_path else RUN_EVALUATION_LOG_DIR
    log_dir = base_log_dir / run_id / agent_name / model_nickname / instance_id

    # Set up logger
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / LOG_INSTANCE
    logger = setup_logger(instance_id, log_file)

    # Run the instance
    container = None
    try:
        # Build + start instance container
        # Mount log_dir to /logs in the container
        # Use mounts instead of volumes to avoid issues with colons in paths
        container = build_container(
            test_spec, 
            client, 
            run_id, 
            logger, 
            rm_image, 
            force_rebuild,
            container_kwargs={'mounts': [Mount(target='/logs', source=str(log_dir.absolute()), type='bind')]}
        )
        container.start()
        logger.info(f"Container for {instance_id} started: {container.id}")
        
        # Fix ownership of /logs to match host user (avoid root-owned files on host)
        host_uid = os.getuid()
        host_gid = os.getgid()
        logger.info(f"Setting /logs ownership to {host_uid}:{host_gid}")
        chown_result = container.exec_run(
            f"chown -R {host_uid}:{host_gid} /logs",
            user="root"
        )
        if chown_result.exit_code != 0:
            logger.warning(f"Failed to change /logs ownership: {chown_result.output.decode(UTF8)}")

        # Get repo and base_commit from instance
        repo = instance.get('repo', test_spec.repo)
        base_commit = instance.get('base_commit', '')
        
        logger.info(f"Repository: {repo}")
        logger.info(f"Base commit: {base_commit}")
        
        # Reset git repository to clean state
        logger.info("Resetting git repository...")
        reset_result = container.exec_run(
            "git reset --hard",
            workdir=DOCKER_WORKDIR,
            user=DOCKER_USER
        )
        if reset_result.exit_code != 0:
            logger.error(f"Failed to git reset --hard: {reset_result.output.decode(UTF8)}")
            raise Exception(f"Failed to git reset --hard: {reset_result.output.decode(UTF8)}")
        else:
            logger.info("Git repository reset successfully")
        
        # Remove all git remotes
        logger.info("Removing git remotes...")
        remove_remotes_result = container.exec_run(
            'bash -c \'for remote_name in $(git remote); do git remote remove "${remote_name}"; done\'',
            workdir=DOCKER_WORKDIR,
            user=DOCKER_USER
        )
        if remove_remotes_result.exit_code != 0:
            logger.error(f"Failed to remove git remotes: {remove_remotes_result.output.decode(UTF8)}")
            raise Exception(f"Failed to remove git remotes: {remove_remotes_result.output.decode(UTF8)}")
        else:
            logger.info("Git remotes removed successfully")
        
        # Checkout to base commit
        if base_commit:
            logger.info(f"Checking out to base commit: {base_commit}")
            checkout_result = container.exec_run(
                f"git checkout {base_commit}",
                workdir=DOCKER_WORKDIR,
                user=DOCKER_USER
            )
            if checkout_result.exit_code != 0:
                logger.warning(f"Failed to checkout to base commit: {checkout_result.output.decode(UTF8)}")
            else:
                logger.info(f"Successfully checked out to {base_commit}")
        
        # Install dependencies with pip
        logger.info("Installing dependencies with pip...")
        pip_install_result = container.exec_run(
            "pip install -e .",
            workdir=DOCKER_WORKDIR,
            user=DOCKER_USER
        )
        if pip_install_result.exit_code != 0:
            logger.warning(f"pip install returned non-zero exit code: {pip_install_result.output.decode(UTF8)}")
        else:
            logger.info("Dependencies installed successfully")
        
        # Install the agent using its install method
        logger.info(f"Installing agent: {agent_name}")
        install_success = agent.install(container, logger, DOCKER_WORKDIR, DOCKER_USER)
        if not install_success:
            logger.error(f"Failed to install agent: {agent_name}")
            raise Exception(f"Agent installation failed for {agent_name}")
        
        # Create the agent execution script from template
        agent_cli_command = agent.get_cli_command()
        agent_script = AGENT_SCRIPT_TEMPLATE.format(agent_command=agent_cli_command)
        
        # Write script to file and copy to container
        script_file = Path(log_dir / "run_agent.sh")
        script_file.write_text(agent_script)
        logger.info(f"Agent script written to {script_file}")
        
        # Copy script to container
        copy_to_container(container, script_file, PurePosixPath("/run_agent.sh"))
        logger.info("Agent script copied to container")
        
        # Make script executable and run it
        chmod_result = container.exec_run(
            "chmod +x /run_agent.sh",
            workdir=DOCKER_WORKDIR,
            user=DOCKER_USER
        )
        if chmod_result.exit_code != 0:
            logger.warning(f"Failed to make script executable: {chmod_result.output.decode(UTF8)}")
        
        # Run the agent script
        logger.info(f"Executing agent script for {agent_name}")
        agent_output, timed_out, total_runtime = exec_run_with_timeout(
            container,
            "/bin/bash /run_agent.sh",
            timeout
        )
        
        # Write agent output to logs
        agent_output_path = log_dir / "agent_output.txt"
        logger.info(f'Agent runtime: {total_runtime:_.2f} seconds')
        with open(agent_output_path, "w") as f:
            f.write(agent_output)
            logger.info(f"Agent output for {instance_id} written to {agent_output_path}")
            if timed_out:
                f.write(f"\n\nTimeout error: {timeout} seconds exceeded.")
                logger.error(f"Agent timed out after {timeout} seconds.")
                raise Exception(f"Agent execution timed out after {timeout} seconds")
        
        # Fix ownership of /logs again after agent execution (some agents create folders during execution)
        logger.info(f"Setting /logs ownership to {host_uid}:{host_gid} after agent execution")
        chown_result = container.exec_run(
            f"chown -R {host_uid}:{host_gid} /logs",
            user="root"
        )
        if chown_result.exit_code != 0:
            logger.warning(f"Failed to change /logs ownership: {chown_result.output.decode(UTF8)}")
        
        # Generate git patch following the correct process
        logger.info("Generating git patch from changes...")
        
        # Configure git pager
        logger.info("Configuring git pager...")
        result = container.exec_run(
            "git config --global core.pager ''",
            workdir=DOCKER_WORKDIR,
            user=DOCKER_USER
        )
        if result.exit_code != 0:
            logger.warning(f"Failed to configure git pager: {result.output.decode(UTF8)}")
        
        # Find and remove any .git directories in subdirectories
        logger.info("Checking for git repositories in subdirectories...")
        result = container.exec_run(
            'find . -type d -name .git -not -path "./.git"',
            workdir=DOCKER_WORKDIR,
            user=DOCKER_USER
        )
        if result.exit_code == 0:
            git_dirs = [p.strip() for p in result.output.decode(UTF8).strip().split('\n') if p.strip()]
            if git_dirs:
                logger.info(f"Found {len(git_dirs)} git directories to remove")
                for git_dir in git_dirs:
                    logger.info(f"Removing {git_dir}")
                    remove_result = container.exec_run(
                        f'rm -rf "{git_dir}"',
                        workdir=DOCKER_WORKDIR,
                        user=DOCKER_USER
                    )
                    if remove_result.exit_code != 0:
                        logger.warning(f"Failed to remove {git_dir}: {remove_result.output.decode(UTF8)}")
        
        # Add all files to git
        logger.info("Adding all files to git staging...")
        result = container.exec_run(
            "git add -A",
            workdir=DOCKER_WORKDIR,
            user=DOCKER_USER
        )
        if result.exit_code != 0:
            logger.warning(f"Failed to git add -A: {result.output.decode(UTF8)}")
        else:
            logger.info("Files staged successfully")
        
        # Remove binary files from git staging
        logger.info("Removing binary files from git staging...")
        remove_binary_cmd = (
            "git diff --cached --numstat | "
            "awk '$1 == \"-\" && $2 == \"-\" {print $3}' | "
            "xargs -r git reset HEAD --"
        )
        result = container.exec_run(
            f'bash -c "{remove_binary_cmd}"',
            workdir=DOCKER_WORKDIR,
            user=DOCKER_USER
        )
        if result.exit_code != 0:
            logger.warning(f"Failed to remove binary files: {result.output.decode(UTF8)}")
        
        # Generate git diff with retries
        git_patch = None
        base_commit = instance.get('base_commit', '')
        n_retries = 0
        max_retries = 5
        
        while n_retries < max_retries and git_patch is None:
            logger.info(f"Generating git diff (attempt {n_retries + 1}/{max_retries})...")
            result = container.exec_run(
                f'bash -c "git diff --no-color --cached {base_commit} > /tmp/patch.diff"',
                workdir=DOCKER_WORKDIR,
                user=DOCKER_USER
            )
            n_retries += 1
            
            if result.exit_code == 0:
                # Read the patch file
                read_result = container.exec_run(
                    "cat /tmp/patch.diff",
                    workdir=DOCKER_WORKDIR,
                    user=DOCKER_USER
                )
                if read_result.exit_code == 0:
                    git_patch = read_result.output.decode(UTF8, errors='replace')
                    logger.info(f"Successfully generated git patch ({len(git_patch)} bytes)")
                    break
                else:
                    logger.warning(f"Failed to read patch file: {read_result.output.decode(UTF8)}")
            else:
                logger.warning(f"Failed to generate git diff: {result.output.decode(UTF8)}")
                if n_retries < max_retries:
                    logger.info("Retrying in 10 seconds...")
                    import time
                    time.sleep(10)
        
        # Save the generated patch
        if git_patch:
            patch_file = log_dir / "generated_patch.diff"
            patch_file.write_text(git_patch)
            logger.info(f"Generated patch saved to {patch_file}")
        else:
            logger.error("Failed to generate git patch after all retries")
            raise Exception(f"Failed to generate git patch after {max_retries} retries")
        
        logger.info(f"Agent execution completed for {instance_id}")
        
        # Write result to output file immediately with locking
        if output_file and file_lock:
            output_record = {
                "instance_id": instance_id,
                "model_name_or_path": agent.__class__.__name__,
                "full_output": "",
                "model_patch": git_patch if git_patch else ""
            }
            
            with file_lock:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'a') as f:
                    f.write(json.dumps(output_record) + '\n')
                logger.info(f"Result written to {output_file}")
        
        return instance_id, {
            "success": True, 
            "runtime": total_runtime, 
            "timed_out": timed_out,
            "patch": git_patch if git_patch else ""
        }
    except Exception as e:
        error_msg = (f"Error in running agent for {instance_id}: {e}\n"
                     f"{traceback.format_exc()}\n"
                     f"Check ({logger.log_file}) for more information.")
        logger.error(error_msg)
        
        # Write error result to output file with locking
        if output_file and file_lock:
            output_record = {
                "instance_id": instance_id,
                "model_name_or_path": agent.__class__.__name__,
                "full_output": str(e),
                "model_patch": ""
            }
            
            with file_lock:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'a') as f:
                    f.write(json.dumps(output_record) + '\n')
        
        raise e
    finally:
        # Remove instance container + image, close logger
        cleanup_container(client, container, logger)
        if rm_image:
            remove_image(client, test_spec.instance_image_key, logger)
        close_logger(logger)


def run_instances(
        instances: list,
        agent_name: str,
        cache_level: str,
        clean: bool,
        force_rebuild: bool,
        max_workers: int,
        run_id: str,
        timeout: int,
        output_file: str = None,
        namespace: str = None,
        instance_image_tag: str = 'latest',
        input_text_key: str = 'problem_statement',
        model_name_or_path: str = None,
        logs_base_path: str = None,
    ):
    """
    Run all instances with the agent in parallel.

    Args:
        instances (list): List of instances
        agent_name (str): Name of the agent CLI tool to run
        cache_level (str): Cache level
        clean (bool): Clean images above cache level
        force_rebuild (bool): Force rebuild images
        max_workers (int): Maximum number of workers
        run_id (str): Run ID
        timeout (int): Timeout for running agent
        output_file (str): Output JSONL file path for patches
        namespace (str): Docker namespace
        instance_image_tag (str): Tag for instance images
        input_text_key (str): Key in instance dict that contains the problem statement
        model_name_or_path (str): Model name or path to use
        logs_base_path (str): Base path to prepend to log directory
    """
    client = docker.from_env()
    test_specs = list(map(
        lambda instance: make_test_spec(instance, namespace=namespace, instance_image_tag=instance_image_tag),
        instances
    ))

    # build environment images
    build_env_images(client, instances, force_rebuild, max_workers)

    # print number of existing instance images
    instance_image_ids = {x.instance_image_key for x in test_specs}
    existing_images = {
        tag for i in client.images.list(all=True)
        for tag in i.tags if tag in instance_image_ids
    }
    if not force_rebuild and len(existing_images):
        print(f"Found {len(existing_images)} existing instance images. Will reuse them.")

    # Create file lock for thread-safe writing
    file_lock = threading.Lock() if output_file else None
    
    # Clear output file if it exists (start fresh)
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Create empty file or truncate existing one
        output_path.touch()
        print(f"Output will be written to: {output_file}")
    
    # Determine model nickname for output file naming
    model_nickname = model_name_or_path
    if "checkpoint" in Path(model_name_or_path).name:
        model_nickname = Path(model_name_or_path).parent.name
    else:
        model_nickname = Path(model_name_or_path).name

    # run instances in parallel
    payloads = []
    for i, test_spec in enumerate(test_specs):
        # Create agent instance for this instance
        agent = get_agent(agent_name, instances[i], input_text_key, model_name_or_path)
        
        payloads.append((
            test_spec,
            instances[i],  # Pass the full instance dict
            agent,  # Pass the agent instance
            should_remove(
                test_spec.instance_image_key,
                cache_level,
                clean,
                existing_images,
            ),
            force_rebuild,
            client,
            run_id,
            model_nickname,
            timeout,
            output_file,
            file_lock,
            logs_base_path,
        ))
    
    # run instances in parallel (results are written incrementally by each worker)
    print(f"Running {len(instances)} instances with agent {agent_name}...")
    results = run_threadpool(run_instance, payloads, max_workers)
    print(f"All instances completed. Results written to {output_file}" if output_file else "All instances completed.")

def run_agent(
    agent_name,
    test_dataset,
    model_name_or_path,
    output_file,
    model_args,
    existing_ids,
    max_cost,
    input_text,
    num_proc,
    logs_base_path=None,
):
    """
    Run agent on all instances in the test dataset.
    
    Args:
        agent_name (str): Name of the agent CLI tool to run
        test_dataset: Dataset containing instances
        model_name_or_path (str): Model name or path (not used in agentic mode)
        output_file (str): Output file path
        model_args: Model arguments (not used in agentic mode)
        existing_ids: Set of already completed instance IDs
        max_cost: Maximum cost (not used in agentic mode)
        input_text (str): Input text field name
        num_proc (int): Number of parallel processes
        logs_base_path (str): Base path to prepend to log directory
    """
    # Filter out already completed instances
    if existing_ids:
        test_dataset = [inst for inst in test_dataset if inst.get(KEY_INSTANCE_ID) not in existing_ids]
        print(f"Filtered dataset to {len(test_dataset)} instances (excluding {len(existing_ids)} already completed)")
    
    run_instances(
        instances=test_dataset,
        agent_name=agent_name,
        cache_level="instance_image",
        clean=False,
        force_rebuild=False,
        max_workers=num_proc,
        run_id="agentic_run",
        timeout=3600,
        output_file=output_file,
        namespace=None,
        instance_image_tag="latest",
        input_text_key=input_text,
        model_name_or_path=model_name_or_path,
        logs_base_path=logs_base_path,
    )
