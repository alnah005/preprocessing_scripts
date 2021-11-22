import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np
import json

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            if type(v) == str:
                v = v.encode()
            txn.put(k.encode(), v)


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            # print(imagePath)
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)

def removeSquareBracketKeywords(text):
    keywordsFound = []
    current_keyword = ""
    openBracket = False
    for i in text:
        if i == "[" and not(openBracket):
            openBracket = True
            continue
        elif i == "]" and openBracket:
            openBracket = False
            keywordsFound.append(current_keyword)
            current_keyword = ""
            continue
        if openBracket:
            current_keyword += i
    for key in keywordsFound:
        if "["+key+"]" in text and "[/"+key+"]" in text:
            text = text.replace("["+key+"]","")
            text = text.replace("[/"+key+"]","")
    return text

if __name__ == '__main__':
    cwd = os.getcwd()

    required_files = {
        'annotations_0': cwd+"/final_annotations_10_test_cropped.json",
        'annotations_1': cwd+"/final_annotations_10_train_cropped.json",
        'annotations_2': cwd+"/final_annotations_10_val_cropped.json",
    
        'output_path_0': cwd+"/../../text_recognition/data/ASM/lmdb_test_no_bracket/",
        'output_path_1': cwd+"/../../text_recognition/data/ASM/lmdb_train_no_bracket/",
        'output_path_2': cwd+"/../../text_recognition/data/ASM/lmdb_val_no_bracket/",
    }
    for i in required_files.values():
        assert os.path.exists(i)


    annotations = {}
    for k in required_files.keys():
        if 'annotations_' in k[:12]:
            assert '.json' in required_files[k][-5:]
            assert 'output_path_'+k[12:] in required_files.keys()
            with open(required_files[k]) as json_file:
                annotations[k] = json.load(json_file)
    for k in annotations.keys():
        output_path = required_files['output_path_'+k[12:]]
        image_paths = []
        texts = []
        for annot in annotations[k]:
            try:
                assert 'im_path' in annot
                assert 'text' in annot
                assert annot['text'] != 'null' and annot['text'] is not None 
                image_paths.append(annot['im_path'])
                texts.append(removeSquareBracketKeywords(annot['text']))
            except:
                print(k,"error in", annot)
        sorted_based_on_text_length = [i[0] for i in sorted(enumerate(texts), key=lambda x:len(x[1]))]
        texts_sorted = [texts[i] for i in sorted_based_on_text_length]
        image_paths_sorted = [image_paths[i] for i in sorted_based_on_text_length]
        
        createDataset(output_path,image_paths,texts)

    
