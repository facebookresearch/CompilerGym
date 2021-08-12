#!/usr/bin/env python3
#
# Script to launch the backend Flask server and frontend server. Usage:
#
#     python run.py
import os
import subprocess
import sys
from time import sleep

os.environ["FLASK_APP"] = "demo_api.py"
p = subprocess.Popen(["flask", "run"])
try:
    for _ in range(5):
        sleep(0.5)
        if p.poll() is not None:
            print("Failed to start python server", file=sys.stderr)
            sys.exit(1)
    os.chdir("frontends/compiler_gym")
    subprocess.check_call(["npm", "install"])
    subprocess.check_call(["npm", "start"])
finally:
    p.terminate()
