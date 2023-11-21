import os
import subprocess

# EXPROOT: directory of experiment:
EXPROOT='/checkpoint/avatar/jinkyuk/flasharray-gridsample-vgg-oct17'

t1dir = os.listdir(EXPROOT)
t2dir=dict()


# sacct -j 134720 --format="jobid,user,account,cluster,Elapsed,start,end"

        # print(" CMD: {}".format(cmd))
        # rproc = subprocess.run(cmd.split(' '), capture_output=True, encoding='utf-8')
        # if not rproc.returncode == 0:
        #     print(cmd)
        # assert rproc.returncode == 0
        # lines = rproc.stdout.split('\n')

jids={}
for exp in t1dir:
    dldirs = os.listdir(f"{EXPROOT}/{exp}")
    #print(dldirs)
    for ed in dldirs:
        edirs = os.listdir(f"{EXPROOT}/{exp}/{ed}")
        #print(edirs)
        for e in edirs:
            efiles= os.listdir(f"{EXPROOT}/{exp}/{ed}/{e}")
            for f in efiles:
                if "sbatch" in f and "err" in f:
                    print(f)
                    jid = f.split('-')[1].split('.')[0]
                    print(jid)
                    if jid not in jids.keys():
                        jids[jid] = f"{EXPROOT}/{exp}/{ed}/{e}"

print(" all jobs found : {}".format(len(jids)))

for k, v in jids.items():
    print(" {} -- {}".format(k, v))
    cmd = f'sacct --job={k} --format=jobid,user,account,cluster,Elapsed,start,end,State'
    #os.system(f"sacct -j {k} --format=\"jobid,user,account,cluster,Elapsed,start,end,State\"")
    #cmd = "ls . "
    #cmd = 'sacct -j 134719 --format=jobid,user,account,cluster,Elapsed,start,end,State'    
    
    print(cmd.split(' '))
    rproc = subprocess.run(cmd.split(' '), capture_output=True, encoding='utf-8')
    #rproc = subprocess.run("sacct -j 134769 --format=\"jobid,user,account,cluster,Elapsed,start,end,State\"", capture_output=True, encoding='utf-8')
    
    if not rproc.returncode == 0:
        #print(cmd)
        print(rproc.returncode)
        
    assert rproc.returncode == 0
    lines = rproc.stdout.split('\n')
    #print(lines)

    for l in lines:
        print(l)
    

