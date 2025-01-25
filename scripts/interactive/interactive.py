import subprocess
import time
import select
import fcntl
import os
import sys
from traceback import print_tb
import click
import pty
import termios
import struct

def run_command(command_string, print_output=False):
    output_lines = []
    try:
        process = subprocess.Popen(
            command_string,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Redirect stderr to stdout
            text=True,
            env=os.environ,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Read and print output in real-time
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                if print_output:
                    print(line.rstrip())  # Print to terminal
                    sys.stdout.flush()  # Ensure output is printed immediately
                output_lines.append(line)  # Store for returning
        
        # Wait for process to complete and get return code
        return_code = process.wait()
        
        if return_code != 0:
            print(f"Command failed with return code: {return_code}")
            return None
            
        return ''.join(output_lines)
        
    except subprocess.SubprocessError as e:
        print(f"Error executing command: {e}")
        return None

def set_non_blocking(fd):
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

def interactive_pipe(command_a, command_b, idle_timeout, filename=None, append=False):
    # If command_b starts with a script name without path, add './'
    if not command_b.startswith(('/', './', '~/')):
        first_space = command_b.find(' ')
        if first_space == -1:
            command_b = './' + command_b
        else:
            command_b = './' + command_b[:first_space] + command_b[first_space:]

    if os.path.exists(filename):
        os.remove(filename)

    # Start command A with a pseudo-terminal
    master_a, slave_a = pty.openpty()

    # Set terminal size
    term_size = os.get_terminal_size()
    winsize = struct.pack("HHHH", term_size.lines, term_size.columns, 0, 0)
    fcntl.ioctl(master_a, termios.TIOCSWINSZ, winsize)

    process_a = subprocess.Popen(command_a, shell=True,
                               stdin=slave_a,
                               stdout=slave_a,
                               stderr=slave_a,
                               close_fds=True)
    os.close(slave_a)

    # Set non-blocking
    set_non_blocking(master_a)

    try:
        while True:
            output = ""
            last_output_time = time.time()

            # Monitor process A's output
            while True:
                ready, _, _ = select.select([master_a], [], [], 0.1)
                current_time = time.time()

                if ready:
                    try:
                        chunk = os.read(master_a, 1024).decode(errors='replace')
                        if chunk:
                            output += chunk
                            last_output_time = current_time
                            sys.stdout.write(chunk)
                            sys.stdout.flush()
                    except OSError:
                        pass

                # Check if process A is still running
                if process_a.poll() is not None:
                    raise EOFError("Process A terminated")

                # Check if we've reached the inactivity timeout
                if (current_time - last_output_time) >= idle_timeout:
                    break

            # Send output to process B if there's any
            if output.strip():
                # Write output to file if filename is specified
                if filename:
                    mode = 'a' if append else 'w'
                    with open(filename, mode) as f:
                        f.write(output)

                # Start a new process B for each interaction
                # process_b = subprocess.Popen(command_b, shell=True,
                                          # stdout=subprocess.PIPE,
                                          # stderr=subprocess.STDOUT,
                                          # stdin=subprocess.PIPE,
                                          # bufsize=1,
                                          # universal_newlines=True,
                                          # cwd=os.getcwd())
                                          #
                try:
                    response = run_command(command_b)

                    # Send input to process B
                    # process_b.stdin.write(output)
                    # process_b.stdin.flush()
                    # process_b.stdin.close()

                    # Get response from process B
                    # response = process_b.stdout.read()
                    if response:
                        # sys.stdout.write(response)
                        # sys.stdout.flush()
                       
                        # Send response back to process A
                        try:
                            if '<ENTER>' in response:
                                response = response.replace('<ENTER>', '')
                                if response.endswith('\n'):
                                    response = response[:-1]
                                if response.endswith('\r'):
                                    response = response[:-1]    
                                if response.endswith('\n'):
                                    response = response[:-1]
                                os.write(master_a, response.encode())
                                os.write(master_a, b'\n')
                            else:
                                os.write(master_a, response.encode())

                        except OSError:
                            raise EOFError("Process A stopped accepting input")
                    else:
                        print('No response from process B')

                    # Wait for process B to complete
                    # process_b.wait()

                except BrokenPipeError:
                    raise EOFError("Process B stopped accepting input")
                finally:
                    pass
                    # Clean up process B
                    # if process_b.poll() is None:
                        # process_b.terminate()
                        # process_b.wait()

    except (KeyboardInterrupt, EOFError) as e:
        print(f"\nTerminating due to: {e}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    finally:
        process_a.terminate()
        process_a.wait()
        os.close(master_a)

@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('command_a', type=str, required=True)
@click.argument('command_b', type=str, required=True)
@click.option('--idle-timeout', '-t', type=float, required=False, default=1.0,
              help='Time in seconds of inactivity before capturing output from command A (default: 1.0)')
@click.option('--filename', '-f', type=str, required=True, default='/tmp/command_output.txt',
              help='File to write command A\'s output to')
@click.option('--append', '-a', is_flag=True, default=False,
              help='Append to the output file instead of overwriting it')
def main(command_a, command_b, idle_timeout, filename, append):
    """
    Run two commands in an interactive pipeline and show output from both.

    COMMAND_A and COMMAND_B are the two commands to run.

    When COMMAND_A stops producing output for IDLE_TIMEOUT seconds:
    1. Capture all of COMMAND_A's output
    2. Write output to file if specified
    3. Send it to COMMAND_B
    4. Take COMMAND_B's response
    5. Send that response back to COMMAND_A

    This cycle repeats until Ctrl+C is pressed.
    """
    interactive_pipe(command_a, command_b, idle_timeout, filename, append)

if __name__ == '__main__':
    main()
