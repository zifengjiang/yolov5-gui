import subprocess
import os
import sys
import tempfile

def run_in_terminal(command):
    # 在macOS中
    if sys.platform == 'darwin':
        with tempfile.NamedTemporaryFile('w', delete=False, suffix='.sh') as f:
            f.write('#!/bin/bash\n')
            f.write(command + '\n')
        os.chmod(f.name, 0o700)
        subprocess.Popen(['open', '-a', 'Terminal.app', f.name])
    # 在Windows中
    elif sys.platform == 'win32':
        subprocess.Popen(['start', 'cmd', '/k', command], shell=True)
    # 在Linux中（需要xterm）
    elif 'linux' in sys.platform:
        subprocess.Popen(['xterm', '-e', command])
    else:
        print("Unsupported platform")

run_in_terminal('conda env list')