import click
import subprocess
import time
import select
import fcntl
import os
import re
import sys

def set_non_blocking(fd):
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

def strip_ansi_codes(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def capture_output(timeout_ms, max_duration_ms, command, delay_ms):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    set_non_blocking(process.stdout.fileno())
    set_non_blocking(process.stderr.fileno())

    # Implement the delay
    if delay_ms > 0:
        time.sleep(delay_ms / 1000)

    output = b''
    last_output_time = time.time()
    start_time = time.time()

    while True:
        ready, _, _ = select.select([process.stdout, process.stderr], [], [], 0.01)

        if ready:
            for stream in ready:
                chunk = stream.read()
                if chunk:
                    output += chunk
                    last_output_time = time.time()

        current_time = time.time()
        
        # Check if we've reached the maximum duration
        if max_duration_ms and (current_time - start_time) * 1000 >= max_duration_ms:
            break

        # Check if we've reached the inactivity timeout
        if (current_time - last_output_time) * 1000 >= timeout_ms:
            break

        if process.poll() is not None:
            break

    process.terminate()
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        process.kill()

    return output.decode('utf-8', errors='replace')

@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('timeout_ms', type=int)
@click.argument('command', nargs=-1, required=True)
@click.option('--max-duration', '-m', type=int, help='Maximum duration in milliseconds before capturing output')
@click.option('--strip-ansi', '-s', is_flag=True, help='Strip ANSI escape codes from output')
@click.option('--delay', '-d', type=int, default=0, help='Delay in milliseconds before starting capture')
def main(timeout_ms, command, max_duration, strip_ansi, delay):
    """
    Capture command output and emit it after a period of inactivity or maximum duration.

    TIMEOUT_MS is the timeout in milliseconds for inactivity.
    COMMAND is the command to capture (use quotes if it contains spaces).
    """
    try:
        captured_output = capture_output(timeout_ms, max_duration, ' '.join(command), delay)
        if strip_ansi:
            captured_output = strip_ansi_codes(captured_output)
        click.echo(captured_output)
    except click.MissingParameter as e:
        click.echo(f"Error: {str(e)}", err=True)
        click.echo(main.get_help(click.get_current_context()), err=True)
        sys.exit(1)

if __name__ == '__main__':
    main()

