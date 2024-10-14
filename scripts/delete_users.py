'''
script deletes users across all hosts listed
'''

import subprocess
import os, sys, random, string

result = subprocess.run(["getent passwd | grep cou | sed \'s/:/ /g\' | awk \'{print $1}\'"], shell=True, capture_output=True, text=True)
users = result.stdout.split('\n')[:-1]

if (len(sys.argv)<2):
    print("Must provide hostnames")
    exit()

hostnames = sys.argv[1].split(',')
for username in users:
    for h in hostnames:
        cmd = f'pdsh -w {h} sudo userdel -r {username}'
        subprocess.run([cmd], shell=True, capture_output=True, text=True)
    cmd = f'rm -rf /scratch/projects/courses001/{username}'
    subprocess.run([cmd], shell=True, capture_output=True, text=True)
    print(f'Removed {username}')
