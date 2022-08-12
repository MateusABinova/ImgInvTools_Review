from asyncio.windows_events import NULL
import numpy as np
import cv2
import os
import json

def getStream(cam = 0):
    return cv2.VideoCapture(cam, cv2.CAP_DSHOW)


def streamConfig(stream , gain =None , exposition = None, fps = None ,bright = None, contrast = None, focus = None, saturation = None ):
    
   
    if stream.isOpened():
        if gain is not None:  
            stream.set(cv2.CAP_PROP_GAIN, gain)        
        if exposition is not None:            
            stream.set(cv2.CAP_PROP_EXPOSURE, exposition)

        if fps is not None:
            stream.set(cv2.CAP_PROP_FPS, fps)

        if focus is not None:
            stream.set(cv2.CAP_PROP_FOCUS , focus)
               
        if saturation is not None:
            stream.set(cv2.CAP_PROP_SATURATION , saturation)
        
        if bright is not None:
            stream.set(cv2.CAP_PROP_BRIGHTNESS , bright)

        if contrast is not None:
            stream.set(cv2.CAP_PROP_CONTRAST , contrast)
    return 
def streamGetConf(stream):
    params = {
        "gain": stream.get(cv2.CAP_PROP_GAIN),
        "exposition": stream.get(cv2.CAP_PROP_EXPOSURE),
        "fps": stream.get(cv2.CAP_PROP_FPS),
        "focus": stream.get(cv2.CAP_PROP_FOCUS),
        "saturation": stream.get(cv2.CAP_PROP_SATURATION),
        "bright": stream.get(cv2.CAP_PROP_BRIGHTNESS),
        "contrast": stream.get(cv2.CAP_PROP_CONTRAST),

    }
    return params




def bitwiseCompare(img1_bw,img2_bw):
    # redimensiona a imagem para o tamanho da primeira
    img2_bw = cv2.resize(img2_bw,(img1_bw.shape[1],img1_bw.shape[0]))

    img_bwa = cv2.bitwise_and(img1_bw, img2_bw)
    img_bwo = cv2.bitwise_or(img1_bw, img2_bw)
    img_bwx = cv2.bitwise_xor(img1_bw, img2_bw)

    return  img_bwx, img_bwo, img_bwa


def binarizeImg(grayImgFrame, modo = 0, threshold = 127, maxVal = 255 ):
    if modo == 1:
        ret, img_bw = cv2.threshold(grayImgFrame,threshold,maxVal,cv2.THRESH_BINARY)
    elif modo == 2:
        img_bw = cv2.adaptiveThreshold(grayImgFrame,maxVal,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    elif modo == 3:
        img_bw = cv2.adaptiveThreshold(grayImgFrame,maxVal,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    else:
        ret,img_bw = cv2.threshold(grayImgFrame,threshold, maxVal, cv2.THRESH_OTSU)
    return img_bw

def saveImg(imgFrame, fileNale="image.png",path = None):
    StandPath = os.getcwd().replace('\\','/')

    if(path is None):
        os.chdir(StandPath)   
   
        return cv2.imwrite(fileNale, imgFrame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    else:       
        teste = os.path.join(os.getcwd(),path)
        os.chdir(teste)
        status = cv2.imwrite(fileNale, imgFrame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        os.chdir(StandPath)  
        return status

def limiarizeBycolor(imgFrame,LoColor, HiColor):    

    hsvImage = cv2.cvtColor(imgFrame, cv2.COLOR_BGR2HSV)
    #marcador pra saber se o pixel pertence ao intervalo ou não
    mask = cv2.inRange(hsvImage, LoColor, HiColor)
    imgFiltered_and = cv2.bitwise_and(imgFrame, imgFrame, mask = mask)    
   

    # Saida Imagem Limiarizada por intervalo de cor
    return imgFiltered_and 

def imageResize(imgFrame, largura = None, altura = None, fx = None):
    """redimensiona imagem por percential ou por um tamanho definido

    Args:
        imgFrame ([vetor]): imagem ou frame vetorizado
        largura ([int], optional): largura que deve ficar a imagem. Defaults to None.
        altura ([int], optional): altura que deve ficar a imagem. Defaults to None.
        fx ([float], optional): proporção da imagem, sendo 1 o tamanho real e 0.1 tanho em 10%.  Defaults to None.

    Returns:
        [vetor]: imagem redimensionada
    """
    try:
        if(largura == None or altura == None and fx != None):
            imgOut = cv2.resize(imgFrame,fx=fx, fy=fx )
        elif(largura != None or altura != None and fx == None):
            imgOut = cv2.resize(imgFrame, (largura, altura))
        else:
            imgOut = None
    except:
        imgOut = None  
    return imgOut

def detectCorner(imgFrame,thresh,blockSize = 2, apertureSize = 3, k = 0.04 ): 

    #  # # Detector parameters
    thresh = thresh * 0.01
    blockSize = blockSize 
    apertureSize = apertureSize
    k = k * 0.0001

    src_gray = cv2.cvtColor(imgFrame, cv2.COLOR_BGR2GRAY)

    src_gray = np.float32(src_gray)
    # Detecting corners    
    dst = cv2.cornerHarris(src_gray,blockSize,apertureSize,k)

    # # Normalizing    
    dst = cv2.dilate(dst,None)

    imgFrame [dst>thresh*dst.max()]=[0,0,255]
    corners = dst>thresh*dst.max()

    resultCorners = np.where(corners == True)

    
    # return the result    
    return imgFrame, resultCorners


def detectCanyBorder(frameImg,threshold1 = None, threshold2 = None, blur = 10 ):
    """realiza a a detecção das borda da imagem e faz blur na imagem conforme parâmetros informados
        obs: aceita apenas imagems em escala de cdinza
    Args:
        frameImg (vetor imagem CV2): [Vetor do frame de video ou imagem CV2]
        threshold1 ([int 0 - 255], optional): [description]. Defaults to None.
        threshold2 ([int 0 - 255], optional): [description]. Defaults to None.
        blur (int, optional): [description]. Defaults to 10.

    Returns:
        [vetor]: imagem das bordas
        [vetor]: imagem original
        [vetor]: imagem blur
    """
    # # Conversão para escala de cinzaq
    # img_gray = cv2.cvtColor(frameImg, cv2.COLOR_BGRA2GRAY)

    # é feito um blur na imagem para melhorar a detecção
    img_blur = cv2.GaussianBlur(frameImg, (0,0),((blur + 0.1)*0.01))
    
    # detecção das bordas (Canny Edge Detection)
    imgEdges = cv2.Canny(image=img_blur, threshold1=threshold1 , threshold2=threshold2) 
    
    return imgEdges , img_blur

def changePerspective(frameImg, imput ):
    """realiza a correção de perspectiva conforme parâmetros informados

    Args:
        frameImg ([vector]): [description]
        imput ([nunpy vector]): [description]

    Returns:
        [vetor]: imagem com correção de perspectiva
        [vetor]: imagem original
    """        
    # height = frameImg.shape[0]  
    # width = frameImg.shape[1]

    width =  round((abs(imput[0][0]- imput[1][0]) + abs(imput[3][0]- imput[2][0]))*0.5)

    height = round(( abs(imput[0][1]- imput[3][1]) + abs(imput[1][1]- imput[2][1]))*0.5)
    

    output = np.float32([[0,0], [width,0], [width,height], [0,height]])   
   
    # converção vetor para NP 
    imput = np.float32(imput)    
    # Tratamento de pontos negativos

    imput.clip(0,)
    # Calcula matriz de perspectiva
    matrix = cv2.getPerspectiveTransform(imput,output)

    # Transfoma ção de perspectiva coloca toda area do imput que está fora em preto 
    imgOutput = cv2.warpPerspective(frameImg, matrix, (width,height), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    return imgOutput

def trackingColor(frameImg, hue, sat, val):
    """faz o rasteamento de um objeto pela cor (indicada pelos parametros)

    Args:
        frameImg (vetor imagem CV2): [Vetor do frame de video ou imagem CV2]
        hue ([mim, max]): [Cor de 0 a 255]
        sat ([mim, max]): [saturação  de 0 a 255]
        val ([mim, max]): [luminosidade de 0 a 255]

    Returns:
        frameImg([vetor]): imagem com retangulo marcando o objeto
        gray([vetor]): máscara aplicada 
        contours([vetor]): posições do contorno
    """
    #transforma a imagem de RGB para HSV
    hsvImage = cv2.cvtColor(frameImg, cv2.COLOR_BGR2HSV)

    #definir os intervalos de cores que vão aparecer na imagem final
    lowerColor = np.array([hue['min'], sat["min"], val["min"]])
    upperColor = np.array([hue['max'], sat["max"], val["max"]])
    
    #marcador pra saber se o pixel pertence ao intervalo ou não
    mask = cv2.inRange(hsvImage, lowerColor, upperColor)
    
    #aplica máscara que "deixa passar" pixels pertencentes ao intervalo, como filtro
    imgFilter = cv2.bitwise_and(frameImg, frameImg, mask = mask)
    
    #aplica limiarização
    imgFilter = cv2.cvtColor(imgFilter, cv2.COLOR_BGR2GRAY)
    _,imgFilter = cv2.threshold(imgFilter, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    #encontra pontos que circundam regiões conexas (contour)
    contours, hierarchy = cv2.findContours(imgFilter, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    #se existir contornos    
    if contours:
        #retornando a área do primeiro grupo de pixels brancos
        maxArea = cv2.contourArea(contours[0])
        contourMaxAreaId = 0
        i = 0
        
        #para cada grupo de pixels branco
        for cnt in contours:
            #procura o grupo com a maior área
            if maxArea < cv2.contourArea(cnt):
                maxArea = cv2.contourArea(cnt)
                contourMaxAreaId = i
            i += 1
            
        #achei o contorno com maior área em pixels
        cntMaxArea = contours[contourMaxAreaId]
        
        #retorna um retângulo que envolve o contorno em questão
        xRect, yRect, wRect, hRect = cv2.boundingRect(cntMaxArea)
        
        #desenha caixa vermelha envolvente com espessura 2
        cv2.rectangle(frameImg, (xRect, yRect), (xRect + wRect, yRect + hRect), (0, 0, 255), 2)        

    return frameImg, imgFilter, contours


def objectDetection( img1, img2, feature = 'sift',nfeatures = None, threshold1 = None, threshold2 = None, nCorrespondencias = 20, fthreshold = None):
    """faz a detecção de uma imagem/objeto dentro da outra
    Args:
        img1 ([type]): [description] Imagem que será procurada
        img2 ([type]): [description] Imagem onde será feita a busca
        feature (str, optional): default 'sift' pode se utilizar sift, orb, akaze, brisk
        nfeatures (int, optional):
        threshold (float, optional): default 'None' 

    Returns:       
        [vetor]: imagem com a area identificada defida por um retangulo
        [vetor]: primeira imagem com a marcação dos pontos computados 
        [vetor]: segunda imagem com a marcação dos pontos computados 
        [vetor]: pontos de marcação da area identificada (contorno) 
    """
    if nCorrespondencias is None:
        nCorrespondencias = 30



    if feature == 'sift':
        detector1 =  cv2.SIFT_create(nfeatures=nfeatures, edgeThreshold=threshold1)
        detector2 =  cv2.SIFT_create(nfeatures=nfeatures, edgeThreshold=threshold2)
        matcher = cv2.BFMatcher(cv2.NORM_L2)
    elif feature == 'orb':
        detector1 =  cv2.ORB_create(nfeatures=nfeatures, edgeThreshold=threshold1, fastThreshold=fthreshold)
        detector2 =  cv2.ORB_create(nfeatures=nfeatures, edgeThreshold=threshold1, fastThreshold=fthreshold)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        # matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    elif feature == 'akaze':
        detector1 = cv2.AKAZE_create(threshold=threshold1 )
        detector2 = cv2.AKAZE_create(threshold=threshold2 )
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    elif feature == 'brisk':
        detector1 = cv2.BRISK_create(thresh=threshold1)
        detector2 = cv2.BRISK_create(thresh=threshold2)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    else:
        detector1 = cv2.AKAZE_create(threshold=threshold1 )
        detector2 = cv2.AKAZE_create(threshold=threshold2 )
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    #computa os descritores
    kp1, desc1 = detector1.detectAndCompute(img1, None)
    kp2, desc2 = detector2.detectAndCompute(img2, None)
    #print('img1 - %d features, img2 - %d features' % (len(kp1), len(kp2)))

    try:
        matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2)

        mkp1, mkp2 = [], []
        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
                m = m[0]
                mkp1.append( kp1[m.queryIdx] )
                mkp2.append( kp2[m.trainIdx] )

        p1 = np.float32([kp.pt for kp in mkp1])
        p2 = np.float32([kp.pt for kp in mkp2])
        kp_pairs = zip(mkp1, mkp2)

   
        H, inliers = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0) 
        # print('%d -> Correspondencias' % (len(inliers)))
     
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        imgOut = np.zeros((h2,w2,3), np.uint8)
        
        # imgOut[:h2, :w2] = img2
        # imgOut = cv2.cvtColor(imgOut, cv2.COLOR_GRAY2BGR)

        if H is not None and len(inliers) > nCorrespondencias:
            contours = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
            contours = np.int32( cv2.perspectiveTransform(contours.reshape(1, -1, 2), H).reshape(-1, 2) + (0,0) )
            # desenha o retangulo no local identificado
            cv2.polylines(img2, [contours], True, (0, 255, 0),1) 
            # print( "Pts contorno",contours)
        else:
            contours = None  

        # if inliers is None:
        #     inliers = np.ones(len(kp_pairs), np.bool_)
    except:
        imgOut = img2
        contours = None

    if(len(kp1) != 0):
        img1Points = cv2.drawKeypoints(img1,kp1,outImage = None,color=(255,0,255), flags=0)
    else:
        img1Points = img1
    if(len(kp2) != 0):
        img2Points = cv2.drawKeypoints(img2,kp2,outImage = None,color=(255,0,255), flags=0)
    else:
        img2Points = img2
    
    return img2,img1Points,img2Points, contours

def checkRef(ref_Path ):
    references = []
    if ref_Path is not None:
            cwd = os.path.join(ref_Path)
            for dir_path, dir_names, file_names in os.walk(cwd):
                for f in file_names:
                    if f.rfind('.jpg')> -1:                  
                        references.append(f)
                        open(f, 'w')                    
                    else:
                        print('Aviso: Não é possivel adicionar o arquivo "' + f +'"')
    else:
        print("Falha: Não foi indicada a pasta de referências")
    return references
def loadJson(ref_Path ,refConf):    
    try:           
        return json.loads(open(ref_Path + '\\'+ refConf, 'r').read())         
    except:
        file1 = open(ref_Path + '\\'+ refConf, 'w')
        file1.write('{"cam":0}')
        file1.close()
        return json.loads(open(ref_Path + '\\'+ refConf, 'r').read()) 

def writeJson(file_Path, dictInfo):    
    with open(file_Path, 'w') as f:
        f.write(json.dumps( dictInfo))

def loadJson(ref_Path ,refConf):    
    try:           
        return json.loads(open(ref_Path + '\\'+ refConf, 'r').read())         
    except:
        if ref_Path == None:
            ref_Path = os.getcwd()
        file1 = open(ref_Path + '\\'+ refConf, 'w')
        file1.write('{"cam":0}')
        file1.close()
        return json.loads(open(ref_Path + '\\'+ refConf, 'r').read()) 

def writeJson(file_Path, dictInfo):
       
    with open(file_Path, 'w') as f:
        f.write(json.dumps( dictInfo))