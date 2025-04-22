import subprocess
import time
import select
import fcntl
import re
import os
import sys
from traceback import print_tb
import click
import pty
import termios
import struct

TOKEN_RE = re.compile(r'(<[^>]+>)', re.IGNORECASE)

SPECIAL_KEYS = {
    # printable controls
    '<ENTER>':      b'\n',
    '<TAB>':        b'\t',
    '<ESC>':        b'\x1b',
    '<BACKSPACE>':  b'\x7f',

    # common control‑letter combinations
    '<CTRL-A>': b'\x01',  '<CTRL-B>': b'\x02',  '<CTRL-C>': b'\x03',
    '<CTRL-D>': b'\x04',  '<CTRL-E>': b'\x05',  '<CTRL-F>': b'\x06',
    '<CTRL-G>': b'\x07',  '<CTRL-H>': b'\x08',  '<CTRL-I>': b'\x09',
    '<CTRL-J>': b'\x0a',  '<CTRL-K>': b'\x0b',  '<CTRL-L>': b'\x0c',
    '<CTRL-M>': b'\x0d',  '<CTRL-N>': b'\x0e',  '<CTRL-O>': b'\x0f',
    '<CTRL-P>': b'\x10',  '<CTRL-Q>': b'\x11',  '<CTRL-R>': b'\x12',
    '<CTRL-S>': b'\x13',  '<CTRL-T>': b'\x14',  '<CTRL-U>': b'\x15',
    '<CTRL-V>': b'\x16',  '<CTRL-W>': b'\x17',  '<CTRL-X>': b'\x18',
    '<CTRL-Y>': b'\x19',  '<CTRL-Z>': b'\x1a',

    # arrows & navigation (ANSI / VT100)
    '<UP>':    b'\x1b[A',
    '<DOWN>':  b'\x1b[B',
    '<RIGHT>': b'\x1b[C',
    '<LEFT>':  b'\x1b[D',
    '<HOME>':  b'\x1b[H',
    '<END>':   b'\x1b[F',
    '<PGUP>':  b'\x1b[5~',
    '<PGDN>':  b'\x1b[6~',
    '<DEL>':   b'\x1b[3~',
    '<INS>':   b'\x1b[2~',

    # Function keys F1‑F12
    '<F1>': b'\x1bOP',  '<F2>': b'\x1bOQ',  '<F3>': b'\x1bOR',  '<F4>': b'\x1bOS',
    '<F5>': b'\x1b[15~','<F6>': b'\x1b[17~','<F7>': b'\x1b[18~','<F8>': b'\x1b[19~',
    '<F9>': b'\x1b[20~','<F10>':b'\x1b[21~','<F11>':b'\x1b[23~','<F12>':b'\x1b[24~',
}

def llm_tokens_to_chunks(text: str) -> list:
    """
    Split TEXT into literal chunks and <TOKENS>, convert each token to the
    appropriate byte sequence, and return the concatenated bytes object.
    Unknown tokens are passed through unchanged (minus the angle‑brackets).
    """
    out_tokens = []
    for chunk in TOKEN_RE.split(text):
        if not chunk:
            continue
        if chunk.startswith('<') and chunk.endswith('>'):
            token = chunk.upper()
            result = SPECIAL_KEYS.get(token, token[1:-1].encode())  # unknown token => use raw text
            out_tokens.append(result)
        else:
            out_tokens.append(chunk)
    return out_tokens

def strip_ansi(text):
    patterns = [
        # Shell prompt and status
        r'%.*\n',  # Clear % lines
        r'\(py[0-9]+\)',  # Python virtual env indicators
        r'0;.*?:',  # Shell title sequences
        r'✚|✱|◼',  # Git status symbols
        r' main ',  # Git branch indicators
        r'❯',  # Shell prompt characters

        # Standard ANSI escape sequences
        r'\x1B(?:[@-Z\\-_]|\[[0-9?]*[ -/]*[@-~])',

        # Terminal hyperlinks - modified to capture and preserve filename
        r'8;;file:\/\/\/.*?/(.*?)8;;',  # File links - replace with captured filename
        r'\x1B]8;;.*?\x1B\\',     # OSC hyperlinks

        # Other terminal-specific sequences
        r'\x1B\][0-9];.*?\x07',   # OSC sequences terminated by BEL
        r'\x1B\][0-9];.*?\x1B\\', # OSC sequences terminated by ST

        # Various control sequences
        r'[\x00-\x08\x0B\x0C\x0E-\x1A\x1C-\x1F\x7F]', # Control characters
        r'\x1B[@-_][0-9:;<=>?]*[-$@-~]',  # CSI and other extended sequences

        # Color codes and formatting
        r'\x1B\[([0-9]{1,2}(;[0-9]{1,2})*)?[m|K]',  # SGR color and format codes
        r'\x1B\[38;5;\d+m',  # 256 color codes
        r'\x1B\[48;5;\d+m',  # 256 background color codes

        # Less common but possible sequences
        r'\x1B%G',          # UTF-8 sequence
        r'\x1B\[(\d+)(;\d+)*m',  # Complex SGR sequences
        r'\x1B\[?[\d;]*[A-Za-z]',  # Catch-all for other CSI sequences

        # Shell-specific cleanup
        r'=llsls>',  # Command artifacts
        r'^\s*$\n'   # Empty lines
    ]

    # First handle the file links separately to preserve filenames
    cleaned = re.sub(r'8;;file:\/\/\/.*?/(.*?)8;;', r'\1', text)

    # Then combine and apply the rest of the patterns
    combined_pattern = '|'.join(p for p in patterns if 'file:///' not in p)
    cleaned = re.sub(combined_pattern, '', cleaned)

    # Clean up any leftover control characters but preserve newlines
    cleaned = re.sub(r'[\x00-\x09\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', cleaned)

    # Clean up multiple blank lines while preserving single newlines
    cleaned = re.sub(r'\n\s*\n', '\n', cleaned)

    return cleaned.strip()

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

        result = ''.join(output_lines)
        return result

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

    if filename and os.path.exists(filename):
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

                output = strip_ansi(output)

                # Write output to file if filename is specified
                if filename:
                    mode = 'a' if append else 'w'
                    with open(filename, mode) as f:
                        f.write(output)

                retry_counter = 0
                while retry_counter < 2:
                    try:
                        response = run_command(command_b)

                        if response:
                            chunks = llm_tokens_to_chunks(response)
                            for chunk in chunks:
                                try:
                                    if isinstance(chunk, bytes):
                                        os.write(master_a, chunk)

                                    else:
                                        os.write(master_a, chunk.encode())
                                except OSError:
                                    raise EOFError("Process A stopped accepting input")
                            break
                        else:
                            print('No response from process B')
                            retry_counter += 1

                    except BrokenPipeError:
                        raise EOFError("Process B stopped accepting input")
                    finally:
                        pass

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
