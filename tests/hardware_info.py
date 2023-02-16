import platform
import subprocess


if platform.system() == "Windows":
    family = platform.processor()
    name = (
        subprocess.check_output(["wmic", "cpu", "get", "name"])
        .decode()
        .strip()
        .split("\n")[1]
    )
    cpu_info = " ".join([name, family])
elif platform.system() == "Linux":
    cpu_info = subprocess.check_output(["lscpu"]).decode()
else:
    cpu_info = platform.processor()

print(f"CPU: {cpu_info}")
