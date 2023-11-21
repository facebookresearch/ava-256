
import cv2 
import os

#ids=['row', 'byj', 'njy']
ids=['vke']

abf=['1.0', '0.1']

cams = ['401036', 
        '401042',
        '401044',
        '401067',
        '401384',
        '401412']

#print(ptemp)
outdir='single-frame-abf1.0-abf0.1-flip'

imagecnt = 40

for id in ids:
    for ca in cams:
        
        outpath=f"{outdir}/{id}/{ca}/"
        os.system(f"mkdir -p {outpath}")

        gts = list()
        maxabf=list()
        minabf=list()
        
        ab = 1.0
        ptemp = "scratch-{abf}-{cam}".format(abf=ab, cam=ca)    
        print(ptemp)
        path=f'{id}/{ptemp}'

        for i in range(imagecnt):
            gfn=f"{path}/{i*2}.jpg"
            maxfn =f"{path}/{i*2+1}.jpg"
            gts.append(gfn)
            maxabf.append(maxfn)


        ab = 0.1
        ptemp2 = "scratch-{abf}-{cam}".format(abf=ab, cam=ca)    
        print(ptemp2)
        path=f'{id}/{ptemp2}'

        for i in range(imagecnt):
            minfn =f"{path}/{i*2+1}.jpg"
            minabf.append(minfn)
                
        #maxabf = sorted(maxabf)
        #gts = sorted(gts)       


        root = '/home/jinkyuk/rsc/neurvol2-jason/helpers/'
        progress=0

        i = 20 
        for idx in range(len(gts)):
            print(" i {} -- {}  -- {} --  {}".format(i, gts[i], maxabf[i], minabf[i]))
            
            x,y,w,h = 30, 60, 600,30

            # Draw black background rectangle
            
            fn = root+gts[i]
            print(" FN : {}".format(fn))
            #img = cv2.imread(f"{fn}")
            #g = img[:,:,:]

            #print("img shape : {}".format(img.shape))            
            #cv2.rectangle(g, (x, x), (x + w, y + h), (0,0,0), -1)            
            #g = cv2.putText(g, f"GT", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            #ofn = f"{outpath}/{progress}.png"
            os.system(f"ln -s {fn} {outpath}/{progress}.jpg")            
            progress += 1
            #cv2.imwrite(ofn, g)
    
            fn = root+maxabf[i]
            print(" FN : {}".format(fn))
            #img = cv2.imread(f"{fn}")
            #g = img[:,:,:]

            #cv2.rectangle(g, (x, x), (x + w, y + h), (0,0,0), -1)            
            #g = cv2.putText(g, f"GT-c{ca}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            #g = cv2.putText(g, f"100% Cams", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

            #cv2.imwrite(f'test-max.jpg', g)
            #ofn = f"{outpath}/{progress}.png"
            os.system(f"ln -s {fn} {outpath}/{progress}.jpg")
            progress += 1
            #cv2.imwrite(ofn, g)


            
            fn = root+minabf[i]
            print(" FN : {}".format(fn))
            #img = cv2.imread(f"{fn}")
            #g = img[:,:,:]

            #cv2.rectangle(g, (x, x), (x + w, y + h), (0,0,0), -1)            
            #g = cv2.putText(g, f"GT-c{ca}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            #g = cv2.putText(g, f"10% Cams", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            #cv2.imwrite(f'test-min.jpg', g)
            #ofn = f"{outpath}/{progress}.png"
            os.system(f"ln -s {fn} {outpath}/{progress}.jpg")            
            progress += 1
            #cv2.imwrite(ofn, g)                    

        cmd= "ffmpeg -framerate 1 -y -i {}/%d.jpg -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" -g 10 -crf 19 ./single-frame-abf1.0-abf0.1-flip/{}-fliptest-gt-100P-10P-cameras-heldout-cam{}.mp4".format(outpath, id, ca)
        os.system(cmd)
