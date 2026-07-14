import os
import time
import urllib3
import requests
import ssl
import socket

# Set global socket timeout to 60 seconds to prevent hanging on network reads
socket.setdefaulttimeout(60.0)

os.environ['KAGGLE_API_TOKEN'] = 'KGAT_6a49c1558a280aaa2f5a9906333e13fe'
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
original_send = requests.Session.send
requests.Session.send = lambda self, request, **kw: original_send(self, request, **{**kw, 'verify': False, 'timeout': 60.0})
ssl._create_default_https_context = ssl._create_unverified_context

from kaggle import api
api.authenticate()

kernel = 'subhageethika123/biohub-3d-cell-tracking-development'
print(f"Monitoring execution of kernel: {kernel}", flush=True)

start_time = time.time()
timeout = 1800  # 30 minutes max wait

while time.time() - start_time < timeout:
    try:
        res = api.kernels_status(kernel)
        status_val = getattr(res, 'status', None)
        status_str = str(status_val).upper()
        failure = getattr(res, 'failureMessage', None)
        
        print(f"Current status: {status_val} (string: {status_str}) | Elapsed: {int(time.time() - start_time)}s", flush=True)
        
        if 'COMPLETE' in status_str:
            print("Execution completed! Attempting submission...", flush=True)
            submit_res = api.competition_submit_code(
                file_name='submission.csv',
                message='Percentile-adaptive sibling-constrained tracker with gap closing and short track filtering (Version 27)',
                competition='biohub-cell-tracking-during-development',
                kernel=kernel,
                kernel_version=27
            )
            print("Submission successful! Response:", submit_res, flush=True)
            if hasattr(submit_res, 'message'):
                print("Message from Kaggle:", submit_res.message, flush=True)
            break
            
        elif 'ERROR' in status_str:
            print("Kernel execution failed with error.", flush=True)
            if failure:
                print("Failure message:", failure, flush=True)
            break
            
        elif 'RUNNING' in status_str or 'QUEUED' in status_str or 'CANCEL_PENDING' in status_str:
            # Still executing/queued/cancelling other runs, continue waiting
            pass
            
        else:
            print(f"Unexpected status type/value: {status_val} ({status_str})", flush=True)
            break
            
    except Exception as e:
        print("Error during check:", e, flush=True)
        
    time.sleep(30)
else:
    print("Monitoring timed out after 30 minutes.", flush=True)
