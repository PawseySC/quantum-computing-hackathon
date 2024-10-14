'''
script produces a bash script to be run to produce users with specific names
and passwords
'''

import subprocess
import os, sys, random, string

if (len(sys.argv)<3):
    print("Must call with a number indicating number of users and hostnames ")
    exit()

nusers = int(sys.argv[1])
hostnames = sys.argv[2].split(',')

# if running command, otherwise just print output
runcommand = True
if (len(sys.argv)==4):
   if (sys.argv[3] == 'n'): runcommand = False

users = []
passwords = []
projects = ['courses001', 'video']
subprocess.run(["set +o history"], shell=True, text=True)

for i in range(1,nusers):
    username = f'cou{i:03d}'
    passwd = ''.join(random.choices(string.ascii_letters, k=13))
    users.append(username)
    passwords.append(passwd)
    for h in hostnames:
        cmds = list()
        cmds.append(f'pdsh -w {h} sudo useradd -m -s /bin/bash {username}')
        cmds.append(f'pdsh -w {h} \'echo \"{username}:{passwd}\" | sudo chpasswd\'')
        for p in projects:
            cmds.append(f'pdsh -w {h} sudo usermod -a -G {p} {username}')
        for c in cmds:
            if runcommand:
                subprocess.run([c], shell=True, capture_output=True, text=True)
            else:
                print(c)
    cmds = list()
    cmds.append(f'sudo mkdir -p /scratch/projects/courses001/{username}')
    cmds.append(f'sudo chgrp courses001 /scratch/projects/courses001/{username}')
    cmds.append(f'sudo chown {username} /scratch/projects/courses001/{username}')
    for c in cmds:
        if runcommand:
            subprocess.run([c], shell=True, capture_output=True, text=True)
        else:
            print(c)

subprocess.run(["set -o history"], shell=True, text=True)

print('Produced users with these passwords')
for i in range(nusers):
    print(users[i], passwords[i])
