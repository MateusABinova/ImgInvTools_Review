import cv2
import inovaToolImage as iti
import numpy as np
import sys
import getopt
import os


def main():
    print("Inova Sistemas Eletrônicos")
    print("Versão 1.0\nDavi Rodriguez")
    opts, imgs = getopt.getopt(sys.argv[1:], 'x', ['ref=','cam=', 'conf=', 'modo=', "dbg="])  
    opts = dict(opts)
    
    try:
        ref_Path = opts.get('--ref', None )
    except:
        ref_Path = None 
    try:
        cam = opts.get('--cam', 0 )
    except:
        cam = 0  
    try:
        modo = opts.get('--modo', '01' )
    except:
        modo = '01'  

    try:        
        stream = iti.getStream(cam)
    except:
        print("Erro: Falha ao identificar a camera") 


    refConf =  "default.json"
    paramsConfig = iti.loadJson(ref_Path, refConf)
    cv2.namedWindow('Ajustes')
    cv2.createTrackbar("exp", 'Ajustes', 60, 100, lambda exp:iti.streamConfig(stream, exposition=(exp * -0.1)))
    cv2.createTrackbar("brig", 'Ajustes', 127, 255, lambda brig:iti.streamConfig(stream, bright=brig ))
    cv2.createTrackbar("fps", 'Ajustes', 30, 60, lambda fps:iti.streamConfig(stream, fps=fps ))
    cv2.createTrackbar("ctrst", 'Ajustes', 500, 1000, lambda cts:iti.streamConfig(stream, contrast=(cts * 0.1)))
    cv2.createTrackbar("sat", 'Ajustes', 127, 255, lambda sat:iti.streamConfig(stream, saturation=sat))
    cv2.createTrackbar("gain", 'Ajustes', 127, 255, lambda gain:iti.streamConfig(stream, gain=gain))
    cv2.createTrackbar("foco", 'Ajustes', 127, 255, lambda focus:iti.streamConfig(stream, focus=focus))
    cv2.namedWindow('cores')
    cv2.createTrackbar("th", 'cores', 0, 255, lambda th:th  )

    if modo == '02':
        cv2.createTrackbar("hueMin", 'cores', 0, 255, lambda x:x)
        cv2.createTrackbar("hueMax", 'cores', 255, 255, lambda x:x)
        cv2.createTrackbar("SatMin", 'cores', 1, 255, lambda x:x)
        cv2.createTrackbar("SatMax", 'cores', 255, 255, lambda x:x)
        cv2.createTrackbar("LumMin", 'cores', 1, 255, lambda x:x)
        cv2.createTrackbar("LumMax", 'cores', 255, 255, lambda x:x)
    
    gerarPadrao(stream, ref_Path, paramsConfig, modo)   
   

   



  


def gerarPadrao(stream, ref_Path, paramsConfig, modo):
    if ref_Path == None:
        ref_Path = os.getcwd()
    points = list()
    ret = None
    cont = 0

    def on_click(event, x, y, p1, p2):
        if event == cv2.EVENT_LBUTTONDOWN:
            cordenadas = (x,y)
            cv2.circle(img=image,center=cordenadas,radius= 3,color=(0,0,255),thickness=-1 )
            points.append(cordenadas) 
            
    while True: 
        if  points.__len__() == 0:
            sts, image = stream.read()
            sts, image2 = stream.read()         
        th     = cv2.getTrackbarPos("th", 'cores',)
        
        if modo == '02':
            hueMin = cv2.getTrackbarPos("hueMin", 'cores')
            hueMax = cv2.getTrackbarPos("hueMax", 'cores')
            SatMin = cv2.getTrackbarPos("SatMin", 'cores')
            SatMax = cv2.getTrackbarPos("SatMax", 'cores')
            LumMin = cv2.getTrackbarPos("LumMin", 'cores')
            LumMax = cv2.getTrackbarPos("LumMax", 'cores')
            LoColor = np.array([hueMin, SatMin, LumMin])
            HiColor = np.array([hueMax, SatMax, LumMax])
            imgFiltered_and = iti.limiarizeBycolor(image,LoColor, HiColor)
            cv2.imshow("image filtered and", imgFiltered_and)  
            imgcutCz = cv2.cvtColor(imgFiltered_and, cv2.COLOR_BGR2GRAY)
            imgBin = iti.binarizeImg(imgcutCz,modo=1,threshold=th)

        else:
            imgcutCz = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
            imgBin = iti.binarizeImg(imgcutCz,modo=1,threshold=th)
            imgBinC =  iti.binarizeImg(image.copy(),modo=1,threshold=th)
            cv2.imshow("imgBinC", imgBinC)

        th = cv2.getTrackbarPos("th", 'cores')
        cv2.imshow("imagem Cam", image)
        cv2.imshow("image Bin", imgBin)
        cv2.setMouseCallback('imagem Cam', on_click)

        if points.__len__() == 4:                
                ret =iti.changePerspective(image2.copy(),points)
                points = list()
                cv2.imshow("Padrao",ret)
                cv2.setWindowProperty('Padrao', cv2.WND_PROP_TOPMOST,1)

        tc = cv2.waitKey(1)
        if tc & 0xFF == ord('s') and ret is not None:
            cont = cont+1   
            imageName =   'Padrao_%d.png'%cont
            confgName =   'Padrao_%d.json'%cont
            iti.saveImg(ret,imageName, ref_Path )
            points = list()
            cv2.destroyWindow("Padrao")
            
            if modo == '02':
                padr = {
                    "refs":[imageName],
                    "tempoTeste":30,
                    "tamMask":6,
                    "metodoDetect":'akaze',
                    "correspondencias":30,
                    "mode":modo,
                    "threshold":th,                
                    "hueMin":hueMin,
                    "hueMax":hueMax,
                    "SatMin":SatMin,
                    "SatMax":SatMax,
                    "LumMin":LumMin,
                    "LumMax":LumMax,               
                    }
            else:
                padr = {
                    "refs":[imageName],
                    "tempoTeste":30,
                    "tamMask":6,
                    "metodoDetect":'akaze',
                    "correspondencias":30,
                    "mode":modo,
                    "threshold":th,                                
                    }
            paramsConfig.update(padr)
            paramsConfig.update(iti.streamGetConf(stream))
            
            iti.writeJson(ref_Path+"\\"+confgName, paramsConfig)  
        

        elif tc & 0xFF == ord('q'):
            print("Finalizado pelo usuário")
            break




main()


