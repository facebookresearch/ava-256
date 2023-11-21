import os

# AIRSTORE_CODEC_AVATAR_20210504_0803_EDL430_FOV_FULL_ORDER_BY_FRAME_CRYPTO

id10=[]#['20211108-1309-ajd691', '20210223-1023-avw368', '20210504-0803-edl430', '20210504-1342-njy448', '20210510-1347-pco068', '20210511-0825-row429', '20210511-1400-ybn667', '20210512-1331-gui516', '20210519-0856-vke790', '20210520-1343-byj247']

root='/checkpoint/avatar/jinkyuk/read-only/ablation-test/26id-ablation-plans'

fns = os.listdir(root)

ids=dict()
for fn in fns:
    if 'GHS' not in fn:
        continue
    dname = fn.split('.')[0]
    print(dname)

    toks = dname.split('--')

    date=toks[1]
    time=toks[2]
    sid=toks[3]    
    #'m--20210405--1000--ROW429--GHS'
    
    tmp=f'{date}-{time}-{sid.lower()}'
    ids[tmp] = 0


files = ids.keys()
print(files)


idlist = ''
clist = ''

for k in files:
    if k not in id10:
        print(k)
        date, time, sid = k.split('-')        
        crypto = f'AIRSTORE_CODEC_AVATAR_{date}_{time}_{sid.upper()}_FOV_FULL_ORDER_BY_FRAME_CRYPTO'
        print(crypto)
        idlist += k
        idlist += ' '
        clist += crypto
        clist += ' '
        wfd = open(k, 'wt')
        wfd.write(f'CODEC_AVATAR_{date}_{time}_{sid.upper()}_FOV_FULL_ORDER_BY_FRAME'.lower())
        wfd.close()

print(idlist)
print(clist)
        

#codec_avatar_20211108_1309_ajd691_fov_full_order_by_frame

#20210224-1237-rto934
