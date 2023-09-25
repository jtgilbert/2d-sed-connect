from typing import List
import time
import subprocess
import logging

def run_subprocess(cwd: str, cmd: List[str]):
    # from github.com/Riverscapes/riverscapes-tools/blob/master/lib/commons/rscommons/hand.py#L124

    start_time = time.time()
    # Realtime logging form subprocess
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)

    for output in iter(process.stdout.readline, b''):
        for line in output.decode('utf-8').split('\n'):
            if len(line) > 0:
                logging.info(line)

    for errout in iter(process.stderr.readline, b''):
        for line in errout.decode('utf-8').split('\n'):
            if len(line) > 0:
                logging.info(line)

    retcode = process.poll()
    if retcode is not None and retcode > 0:
        logging.error(f'Process returned with code {retcode}')

    ellapsed_time = time.time() - start_time
    logging.info(f'Command completed in {ellapsed_time}')

    return retcode
