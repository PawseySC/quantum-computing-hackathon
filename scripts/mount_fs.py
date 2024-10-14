'''
script produces a bash script to be run to produce users with specific names
and passwords
'''

import subprocess
import os, sys, random, string

if (len(sys.argv)<2):
    print("Must call with  hostnames ")
    exit()

hostnames = sys.argv[1].split(',')

# if running command, otherwise just print output
runcommand = True
if (len(sys.argv)==3):
   if (sys.argv[2] == 'n'): runcommand = False

#sudo mount -t lustre -o relatime,flock fs-00e6a3445930a4ac2.fsx.ap-southeast-2.amazonaws.com@tcp:/ept43bmv
#sudo mount -t nfs4 -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport fs-030d7ae6a9f18c91e.efs.ap-southeast-2.amazonaws.com:/ /group/

mounts = {
    "name": [
        "lustre",
        "efs as nfs4",
    ],
    "ip" : [
        "fs-00e6a3445930a4ac2.fsx.ap-southeast-2.amazonaws.com@tcp:/ept43bmv",
        "fs-030d7ae6a9f18c91e.efs.ap-southeast-2.amazonaws.com:/",
     ],
    "mountpoint" : [
        "/scratch",
        "/group",
    ],
    "args" : [
        "-t lustre -o relatime,flock",
        "-t nfs4 -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=30,retrans=2,noresvport",
    ],

for i in range(len(mounts['name'])):
    n, ip, mp, a = mounts['name'][i], mounts['ip'][i], mounts['mountpoint'][i], mounts['args'][i]
    print(f'Mounting {n}')
    for h in hostnames:
        cmd = f'pdsh -w {h} sudo mount {a} {ip} {mp}'
        if runcommand:
            subprocess.run([cmd], shell=True, capture_output=True, text=True)
        else:
            print(cmd)
