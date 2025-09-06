import time
import os

print("Starting CPU test job.")
print(f"Job running on host: {os.uname()}")

for i in range(10):
    print(f"Processing step {i+1}/10...")
    time.sleep(1)

print("CPU test job finished successfully.")
