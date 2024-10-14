#!/bin/bash

# update default packages 
sudo apt update 
sudo apt -y --no-install-recommends install python3-pip python3-virtualenv python3-venv jupyterhub pdsh



# clone appropriate repo
git clone https://github.com/PawseySC/pawsey-aws-config 
# set requirements 
# these are based on having pennylane qiskit qmuvi seaborn matplotlib jupyterhub installed
export qhackreqs=pawsey-aws-config/qhack/python/requirements.txt

# this would require setting up the system to accept no keys and configure the ssh state
# create the users 
# python pawsey-aws-config/scripts/create_jupyterhub_users.py 2 $(hostname) > userlist.txt 

# create virtual environment 
python3 -m venv qhack-env
source qhack-env/bin/activate 
pip install -r ${qhackreqs} 
deactivate 

# run "The littlest jupyter hub" script
# the script will setup a jupyter hub with the following 
wget https://tljh.jupyter.org/bootstrap.py 
sudo python3 bootstrap.py \
--user-requirements-txt-url ${qhackreqs} \
--admin qhack-admin:footest1 \
--show-progress-page 

# setup https, does require setting up certs
# sudo tljh-config set https.enabled true
# sudo tljh-config set https.letsencrypt.email learn@pawsey.org.au
# sudo tljh-config add-item https.letsencrypt.domains learn.pawsey.org.au
# sudo tljh-config reload proxy

# lets add users 
python3 pawsey-aws-config/
sudo tljh-config add-item users.admin

# now copy templates to the juptyer up region 
cp -r pawsey-aws-config/qhack/templates #need destination directory