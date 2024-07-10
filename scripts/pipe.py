import pyperclip
import os
import glob

def get_latest_pipe():
    # Get all pipes matching the pattern
    pipes = glob.glob('/tmp/llmvm_client_pipe_*')
    
    if not pipes:
        raise FileNotFoundError("No matching pipes found in /tmp")
    
    # Sort pipes by modification time (most recent first)
    latest_pipe = max(pipes, key=os.path.getmtime)
    
    return latest_pipe

def send_to_pipe(pipe_path, content):
    with open(pipe_path, 'w') as pipe:
        pipe.write(content)

def main():
    try:
        # Get clipboard content
        clipboard_content = pyperclip.paste()
        
        # Get the latest pipe
        latest_pipe = get_latest_pipe()
        
        # Send clipboard content to the pipe
        send_to_pipe(latest_pipe, clipboard_content)
        
        print(f"Successfully sent clipboard content to {latest_pipe}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
