import platform
import subprocess


if platform.system() == "Windows":
    family = platform.processor()
    name = subprocess.check_output(["wmic","cpu","get", "name"]).decode().strip().split("\n")[1]
    print("CPU: ", ' '.join([name, family]))
else:
    print(f"CPU: {platform.processor()}")
