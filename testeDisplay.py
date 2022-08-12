from asyncio.windows_events import NULL
from distutils.log import debug
from pickletools import read_unicodestring1
import cv2
import inovaToolImage as iti
import numpy as np
import sys
import getopt
import time

def main():    
    print("Inova Sistemas Eletrônicos")
    print("Davi Rodriguez - Versão 1.0")
    opts, cfg = getopt.getopt(sys.argv[1:], 'x', ['ref=', 'conf=', "modo=", "dbg="])  
    opts = dict(opts)

    global dbg
    try:
        ref_Path = opts.get('--ref', None )
    except:
        ref_Path = None    
    try:
        refConf = opts.get('--conf', None )
    except:
        refConf = None
    try:
        modo = opts.get('--modo', None )
    except:
        modo = None
    try:
        dbg = opts.get('--dbg', 'n' )
    except:
        dbg = 'n'
        
    if ref_Path is None:
        print("Erro:  Falha na configuração, Path reference")
        paramsConfig = None
    elif refConf is None:
        print("Erro: Falha na configuração, arquivo Json")
        paramsConfig = None
    else:      

        selModo(ref_Path, refConf,modo, 0 )

        #try:        
        #    stream = iti.getStream(paramsConfig.get("cam"))
        #except:
        #    stream = None
        #    print("Erro: Falha ao identificar a camera")
        #
        #if stream is not None:
        #    selModo()
        #    
        #else: 
        #    print("Erro: Modo de de validação inválido")

def selModo(ref_Path, refConf, modo, cfgi):

    vectrefConf = refConf.replace(' ','').split(';')
    
    paramsConfig = iti.loadJson(ref_Path, vectrefConf[cfgi])

    stream = iti.getStream(paramsConfig.get("cam",0))

    if modo == '01':
        configStream(stream, paramsConfig) 
        modo01(stream, ref_Path, paramsConfig)
    elif modo == '02':        
        configStream(stream, paramsConfig) 
        modo02(stream, ref_Path, paramsConfig)
    elif modo == '03':        
        configStream(stream, paramsConfig) 
        modo03(stream, ref_Path, paramsConfig)
        pass
    
 

def modo01(stream,ref_Path,paramsConfig):
    star_time = time.time()
    refs = paramsConfig.get("refs")
    th =  paramsConfig.get("threshold")
    feature = paramsConfig.get("metodoDetect")
    nCrpdc = paramsConfig.get("correspondencias")
    tamMask = paramsConfig.get("tamMask")
    tempoTeste = int(paramsConfig.get("tempoTeste"))

    if len(refs) != 0:
        validated = np.zeros(len(refs), dtype=bool)
        ind = 0
        mask = np.full((tamMask,tamMask),255,np.uint8) 

        while True:            
            success, imgFrame = stream.read()
            if success == False:
                time.sleep(0.3)
                success, imgFrame = stream.read()
            if success:
                
                if validated[ind] == False:

                    imgRef= cv2.imread(ref_Path +"/"+ refs[ind])
                    
                    # cv2.imshow("imagem ref",imgRef)

                    imgOut, img1Points, img2Points, Contours = iti.objectDetection(img1=imgRef,img2= imgFrame.copy(),feature=feature, nCorrespondencias = nCrpdc)

                    cv2.imshow("imagem cam",imgFrame)  
                   
                    if dbg  == 'y':
                        cv2.imshow("imagem Points 1",img1Points)
                        cv2.imshow("imagem Points 2",img2Points)
                    
                    
                    if Contours is not None:
                        cutImg = iti.changePerspective(imgFrame, Contours )
                        

                        cutImgCz = cv2.cvtColor(cutImg, cv2.COLOR_BGR2GRAY)
                        cutImgBin =  iti.binarizeImg(cutImgCz,modo=1,threshold=th)

                        refCz = cv2.cvtColor(imgRef, cv2.COLOR_BGR2GRAY)
                        refBin =  iti.binarizeImg(refCz,modo=1,threshold=th)
                        compXorImg, _ , _ = iti.bitwiseCompare(refBin,cutImgBin)


                        if dbg  == 'y':
                            cv2.imshow("imagem recortada",cutImg)
                            cv2.imshow("ref_bin",refBin)
                            cv2.imshow("refCz",refCz)
                            cv2.imshow("cutImgBin",cutImgBin)
                            cv2.imshow("cutImgCz",cutImgCz)

                        cv2.imshow("xor",compXorImg) 

                        result = cv2.matchTemplate(compXorImg,mask,cv2.TM_SQDIFF_NORMED)
                        _minVal, _maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)

                        if _minVal > 0.09 or  _maxVal <= 0: 
                            # iti.saveImg(compXorImg ,'padrao_aprovado_%d.jpg'%ind,'/result' )
                            # b, g, r =  cv2.split(cutImg)
                            # print('Info: Validado %s'%refs[ind])
                            # print('Info:P%d'%ind +' VM = %d'%int(np.median(r)))
                            # print('Info:P%d'%ind +' VD = %d'%int(np.median(g)))
                            # print('Info:P%d'%ind +' AZ = %d'%int(np.median(b)))
                            validated[ind] = True
                            iti.saveImg(compXorImg,refs[ind],ref_Path.replace("reference", "validated"))


                ############## daqui pra baixo está certo ###############       
                ind += 1
                if ind >= len(refs):
                    ind = 0
                
                if validated.max() and validated.min():
                    print("Resultado: Aprovado")
                    print('╰(*°▽°*)╯')                    
                    stream.release()
                    break
                elif int( time.time() - star_time) > tempoTeste:
                    print("Resultado: Reprovado")
                    stream.release()
                    break              

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Resultado: Finalizado pelo usuário")
                    stream.release()
                    break
            else:
                stream.release()
                print("Erro: Sem imagem da camera")
                break  
    else:
        stream.release(stream, )        
        print("Erro: falta de imagem padrão!")     



def modo02(stream,ref_Path,paramsConfig):   
     
    print ('Modo 02 Color Filter')
    star_time = time.time()
    refs = paramsConfig.get("refs")
    th =  paramsConfig.get("threshold")
    hueMin = paramsConfig.get("hueMin")
    hueMax = paramsConfig.get("hueMax")
    SatMin = paramsConfig.get("SatMin")
    SatMax = paramsConfig.get("SatMax")
    LumMin = paramsConfig.get("LumMin")
    LumMax = paramsConfig.get("LumMax")

    feature = paramsConfig.get("metodoDetect")
    nCrpdc = paramsConfig.get("correspondencias")
    tempoTeste = int(paramsConfig.get("tempoTeste"))

    if refs is not None and len(refs) != 0:
        validated = np.zeros(len(refs), dtype=bool)
        ind = 0
        tamMask = 4
        mask = np.full((tamMask,tamMask),255,np.uint8) 


        while True:            
            success, imgFrame = stream.read()
            if success:
                
                if validated[ind] == False:

                    imgRef = cv2.imread(ref_Path +"/"+ refs[ind])
                    
                    # cv2.imshow("imagem ref",imgRef)

                    imgOut, img1Points, img2Points, Contours = iti.objectDetection(img1=imgRef,img2= imgFrame.copy(),feature=feature, nCorrespondencias = nCrpdc)

                    cv2.imshow("imagem cam",imgFrame)  
                   
                    if dbg  == 'y':
                        cv2.imshow("imagem Points 1",img1Points)
                        cv2.imshow("imagem Points 2",img2Points)
                    
                    
                    if Contours is not None:
                        cutImg = iti.changePerspective(imgFrame, Contours )

                        LoColor = np.array([hueMin, SatMin, LumMin])
                        HiColor = np.array([hueMax, SatMax, LumMax])

                        imgFiltered = iti.limiarizeBycolor(cutImg,LoColor, HiColor)
                        
                        #cv2.imshow("image filtered and", imgFiltered)


                        cutImgCz = cv2.cvtColor(imgFiltered, cv2.COLOR_BGR2GRAY)
                        cutImgBin =  iti.binarizeImg(cutImgCz,modo=1,threshold=th)

                        refFiltered = iti.limiarizeBycolor(imgRef,LoColor, HiColor)
                        refCz = cv2.cvtColor(refFiltered, cv2.COLOR_BGR2GRAY)
                        refBin =  iti.binarizeImg(refCz,modo=1,threshold=th)
                        compXorImg, _ , _ = iti.bitwiseCompare(refBin,cutImgBin)
                        
                        if dbg  == 'y':
                            cv2.imshow("imagem refBin",refBin)
                            cv2.imshow("imagem cutImgBin",cutImgBin)
                            cv2.imshow("imagem refFiltered",refFiltered)                       
                            cv2.imshow("imagem recortada",cutImg)

                        cv2.imshow("xor",compXorImg)

                        result = cv2.matchTemplate(compXorImg,mask,cv2.TM_SQDIFF_NORMED)
                        _minVal, _maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)

                        if _minVal > 0.09 or  _maxVal <= 0: 
                            validated[ind] = True
                            iti.saveImg(compXorImg,refs[ind],ref_Path.replace("reference", "validated"))


                ############## daqui pra baixo está certo ###############       
                ind += 1
                if ind >= len(refs):
                    ind = 0
                
                if validated.max() and validated.min():
                    print('╰(*°▽°*)╯')                    
                    print("Resultado: Aprovado")
                   
                    stream.release()
                    break
                elif int( time.time() - star_time) > tempoTeste:
                    print("Resultado: Reprovado")
                    break

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Resultado: Finalizado pelo usuário")
                    stream.release()
                    break
            else:
                stream.release()
                print("Erro: Sem imagem da camera")
                break  
    else:
        stream.release()        
        print("Erro: falta de imagem padrão!")

def configStream(stream, paramsConfig):
    iti.streamConfig(stream,
        gain=paramsConfig.get("gain"),
        exposition=paramsConfig.get("exposition"),
        fps=paramsConfig.get("fps"),
        bright=paramsConfig.get("bright"),
        contrast=paramsConfig.get("contrast"),
        focus=paramsConfig.get("focus"),
        saturation=paramsConfig.get("saturation"),        
        )    
    time.sleep(0.5)    
    return


def modo03(stream,ref_Path,paramsConfig): 
    star_time = time.time()
    th =  paramsConfig.get("threshold")
    BlueValue = None
    GreenValue = None
    RedValue = None
    tempoTeste = int(paramsConfig.get("tempoTeste"))    
    

    while True:            
        success, imgFrame = stream.read()
        if success:
            cv2.imshow("imagem cam",imgFrame)
            

            FrameColorBin =  iti.binarizeImg(imgFrame,modo=1,threshold=th)
            b, g, r =  cv2.split(FrameColorBin)
            cv2.imshow("FrameColorBin cam",FrameColorBin)

            mean_b  = int(np.median(b))
            mean_g = int(np.median(g))
            mean_r = int(np.median(r))          


            if mean_b == 0 and mean_g == 0 and mean_r == 0:               
                print('Info_COLOR: BLACK')
            if mean_b == 255 and mean_g == 0 and mean_r == 0:
                BlueValue =[mean_r,mean_g ,mean_b]
                #print('Info_COLOR: BLUE')
                #iti.saveImg(FrameColorBin,"BLUE",ref_Path.replace("reference", "validated"))
            if mean_b == 0 and mean_g == 255 and mean_r == 0:
                GreenValue =[mean_r,mean_g ,mean_b]
                #print('Info_COLOR: GREEN')
                #iti.saveImg(FrameColorBin,"GREEN",ref_Path.replace("reference", "validated"))
            if mean_b == 0 and mean_g == 0 and mean_r == 255:
                RedValue =[mean_r,mean_g ,mean_b]
                #print('Info_COLOR: RED')
                #iti.saveImg(FrameColorBin,"RED",ref_Path.replace("reference", "validated"))
            if mean_b == 255 and mean_g == 255 and mean_r == 255:                
                print('Info_COLOR: WHITE') 

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Resultado: Finalizado pelo usuário")
                stream.release()
                break
            if BlueValue is not None and RedValue is not None and GreenValue is not None:
                print('COLOR_RED:%s'%RedValue)
                print('COLOR_GREEN:%s'%GreenValue)
                print('COLOR_BLUE:%s'%BlueValue)
                print("Resultado: Aprovado")
                break

            elif int( time.time() - star_time) > tempoTeste:
                print("Resultado: Reprovado")
                break

            
        else:
            stream.release()
            print("Erro: Sem imagem da camera")
            break  
    

main()
