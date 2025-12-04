import cv2
import numpy as np
import copy
import labelme.utils.opencv as ocvutil
from labelme.logger import logger
from labelme.shape import Shape
import torch
import torchvision
from labelme.re3.network import Re3Net
import labelme.re3.bb_util as bb_util
import labelme.re3.im_util as im_util
import labelme.utils.ssdutils as ssdutils
import labelme.utils.ssdmodel as model
import labelme.utils.flagmodel as flagmodel

import random
from qtpy import QtCore

# Optional YOLO11 (ultralytics) backend for auto-annotation
try:
    from ultralytics import YOLO as UltralyticsYOLO
except Exception:
    UltralyticsYOLO = None
    logger.warn("Ultralytics YOLO not available. YOLO11 auto-annotation will be disabled.")


MAX_TRACK_LENGTH = 16
CROP_SIZE = 227
CROP_PAD = 2
SSDSIZE=300
COCOCLASS=['background','person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
SSDMULTIBOX=None
DEVICE=None
RE3MODEL=None
CLASSIFIER=None
CONFIG=None
FLAGCLASS=None
REFIMAGE=None
CURIMAGE=None
TRANSROTATION=None

YOLO11NET = None
YOLO11CLASS = ['background','person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']  # class names for YOLO11 (from ultralytics)


def displayImage(image):
    #im2=(image*127+128).to(torch.uint8).cpu()
    im2=image.cpu()#*127+128).to(torch.uint8).cpu()
    if len(image.shape)==4:
        im2=im2.squeeze(0)
    t=torchvision.transforms.ToPILImage()
    i=t(im2)
    i.show()

class SSD():

    def __init__(self):
        self.device = DEVICE
        self.precision=torch.float32 #16 if torch.cuda.is_available() else torch.float32
        self.model = model.SSD300(backbone=model.ResNet("resnet34"))
        self.model = self.model.to(self.precision)

        def batchnorm_to_float(module):
            """Converts batch norm to FP32"""
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.float()
            for child in module.children():
                batchnorm_to_float(child)
            return module
        if CONFIG["ssdmodel"] is None:
            weights=torch.hub.load_state_dict_from_url('https://keeper.mpdl.mpg.de/f/bd06ac1f6cf84da8b749/?dl=1', map_location=lambda storage, loc: storage, file_name='ssd.pt')['model']
        else:
            weights=torch.load(CONFIG['ssdmodel'], map_location=lambda storage, loc: storage)
            try:
                weights=weights['model']
            except:
                pass
        self.model.load_state_dict(weights)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.hardEmpty = torch.empty((0),dtype=self.precision,requires_grad=False).to(self.device)
        self.hardZero = torch.zeros(1,dtype=torch.long,requires_grad=False).to(self.device)
        self.dboxes = ssdutils.dboxes300_coco()
        self.encoder = ssdutils.Encoder(self.dboxes)
        self.resizer = torchvision.transforms.Resize((SSDSIZE,SSDSIZE),antialias=True,interpolation=torchvision.transforms.InterpolationMode.BICUBIC).to(self.device)


    def inference(self,prediction, min_conf=0.01, nms_iou=0.5):
        loc_p,conf_p = prediction
        loc_p = loc_p.detach().requires_grad_(False)
        conf_p = conf_p.detach().requires_grad_(False) # we don't backpropagate through this

        loc_decoded,conf_p=self.encoder.scale_back_batch(loc_p,conf_p)

        labels = conf_p[:,:,1:] # ignore label 0 (background)

        labelmask = labels >= min_conf

        results=[]
        #nms needs to be done the slow way 
        for frame in range(labels.shape[0]):
            lboxes=[self.hardEmpty.view(0,3,2)]
            for label in range(labels.shape[2]):
                framelabelscores=labels[frame,:,label]
                frameboxes=loc_decoded[frame]
                indices=labelmask[frame,:,label].nonzero().flatten() #this is only needed because of buggy pytorch ONNX export, which doesn't like boolean mask access
                scores = torch.cat((framelabelscores[indices],self.hardZero.to(self.precision)),0)
                boxes = torch.cat((frameboxes[indices],self.hardZero.to(self.precision).view(1,1).expand(1,4)),0)

                # do non maximum suppression
                index = scores.argsort(descending=True)[0:200] # only the 200 highest scores per class are kept, speeds things up in a worst case scenario
                scores=scores[index]
                boxes=boxes[index]
                overlapping = (ssdutils.calc_iou_tensor(boxes,boxes) > nms_iou) # 2x2 boolean matrix of all boxes with too high jaccard overlap - each row has at least one True value on diagonal
                scorematrix = overlapping.float() * scores[:,None]       #this replaces the boolean values with the scores of the column-box

                keep = (scorematrix.max(dim=0)[0] == scores).nonzero().view(-1)

                scores = scores[keep]
                boxes = boxes[keep].view(-1,2,2)
                # this is serialisable for export to ONNX

                score_enc = torch.cat( (scores.unsqueeze(1)*0 + label + 1, scores.unsqueeze(1)), 1).unsqueeze(1)
                boxes = torch.cat( (score_enc,boxes), 1)
                # boxes are now of shape [detections][3][2]
                # with content [[class,conf],[x1,y1],[x2,y2]], ...
                lboxes.append(boxes)
            lboxes = torch.cat( lboxes,0 )
            index = lboxes[:,0,1].sort(descending=True)[1]
            results.append(lboxes[index].contiguous())

        return results

    def prepare(self,image,bbox):
        frame=torch.from_numpy(image).to(self.device).permute(2,0,1)
        frame=frame.to(self.precision)[[2,1,0],:,:] * (1.0/127.0) - 1.0
        H=frame.shape[1]
        W=frame.shape[2]
        w=bbox[2]-bbox[0]
        h=bbox[3]-bbox[1]
        x0=bbox[0]+w/2
        y0=bbox[1]+h/2
        d=h
        if (w>h):
            d=w
        x1=x0-d*float(CONFIG['ssd_track_crop_factor'])*0.5
        x2=x0+d*float(CONFIG['ssd_track_crop_factor'])*0.5
        y1=y0-d*float(CONFIG['ssd_track_crop_factor'])*0.5
        y2=y0+d*float(CONFIG['ssd_track_crop_factor'])*0.5
        x1,y1,x2,y2 = [ (int(a) if a>=0 else 0) for a in [x1,y1,x2,y2]]
        x1,x2 = [ (a if a<=W-1 else W-1) for a in [x1,x2]]
        y1,y2 = [ (a if a<=H-1 else H-1) for a in [y1,y2]]
        subframe=frame[:,y1:y2,x1:x2]
        resized=self.resizer(subframe).unsqueeze(0)
        r0 = self.model(resized)
        r = self.inference(r0)[-1].float()
        r[:,1:,0]*=float(x2-x1)
        r[:,1:,1]*=float(y2-y1)
        r[:,1:,0]+=x1
        r[:,1:,1]+=y1
        candidates=r[:,1:,:].view(-1,4).contiguous()
        candidates=torch.cat((candidates,self.hardZero.float().view(1,1).expand(1,4)),0)
        return candidates

    def detect(self,candidates,bbox):
        bboxtensor=torch.tensor(bbox).to(self.device).view(1,4)
        overlapping = (ssdutils.calc_iou_tensor(bboxtensor,candidates)).view(-1)
        keep=torch.argmax(overlapping)
        logger.info(("best ssd box: %f :"%(overlapping[keep]))+str(candidates[keep]))
        if (overlapping[keep]<float(CONFIG['ssd_re3_correction_min_iou'])):
            logger.info("BOX is below threshold :( ")
            # ssd found nothing
            return None
        else:
            #ssd DID find something
            logger.info("SSD multibox found box "+str(candidates[keep]))
            return [float(candidates[keep,0]), float(candidates[keep,1]), float(candidates[keep,2]), float(candidates[keep,3])]

    def tile(self,image,resolution):
        dim=image.shape[1:]
        cx=int((dim[1]/resolution)*2.5)
        if (cx<2):
            cx=2
        cy=int((dim[0]/resolution)*2.5)
        if (cy<2):
            cy=2
        ix=int((dim[1]-resolution)/(cx-1))
        iy=int((dim[0]-resolution)/(cy-1))
        tiles=[]
        for y in range(cy):
            for x in range(cx):
                tiles.append([x*ix,y*iy,resolution,resolution])
        return tiles

    def alltiles(self,image):
        # top level, inspect whole image
        tiles=[[0,0,image.shape[2],image.shape[1]]]
        # second level, fix aspect ratio
        if image.shape[2]>image.shape[1]:
            tres=int(image.shape[1]*0.95)
        else:
            tres=int(image.shape[2]*0.95)
        # keep reducing 50% - might be skipped for small images
        while tres>(float(CONFIG['ssd_min_resolution'])*2.0):
            tiles=tiles+self.tile(image,tres)
            tres=int(tres*0.5)
        # tile at actual network resolution too
        #tiles=tiles+tile(image,networkres)
        return tiles

    def dotile(self,image,tile):
        #we send a crop of the image through the detector network and collect the results...
        subimage=image[:,tile[1]:tile[1]+tile[3],tile[0]:tile[0]+tile[2]]
        subimage2=self.resizer(subimage).unsqueeze(0)
        
        loc_p,conf_p = self.model(subimage2)
        loc_p = loc_p.detach().requires_grad_(False)
        conf_p = conf_p.detach().requires_grad_(False)
        loc_decoded,conf_p=self.encoder.scale_back_batch(loc_p,conf_p)

        conf_p = conf_p[:,:,1:] # ignore label 0 (background)
        return(loc_decoded,conf_p)

    def findnew(self,image,oldrectangles, min_conf=0.01, nms_iou=0.5):
        frame=torch.from_numpy(image).to(self.device).permute(2,0,1).to(self.precision)[[2,1,0],:,:] * (1.0/127.0) - 1.0
        w=frame.shape[2]
        h=frame.shape[1]
        exclude=torch.cat((torch.tensor(oldrectangles,dtype=self.precision,device=self.device),self.hardZero.to(self.precision).view(1,1).expand(1,4)),0)

        locs=[]
        confs=[]

        aspect=float(w)/float(h)
        if aspect>1.0:
            xfactor=1.0
            yfactor=aspect
        else:
            xfactor=aspect
            yfactor=1.0
        bigframe=torch.zeros((3,SSDSIZE*5,SSDSIZE*5),device=self.device,dtype=self.precision)
        smallframe=self.resizer(frame)  #whole frame to 300x300
        bigframe[:,2*SSDSIZE:3*SSDSIZE,2*SSDSIZE:3*SSDSIZE]=smallframe # insert into here
        for size in [0.9,0.7,0.5]:
            fx=SSDSIZE*xfactor/size
            fy=SSDSIZE*yfactor/size
            x1=int( 2.5*SSDSIZE - 0.5*fx)
            x2=int( 2.5*SSDSIZE + 0.5*fx)
            y1=int( 2.5*SSDSIZE - 0.5*fy)
            y2=int( 2.5*SSDSIZE + 0.5*fy)
            subframe=bigframe[:,y1:y2,x1:x2]
            subframe2=self.resizer(subframe).unsqueeze(0)
            #displayImage(subframe2)
            loc_p,conf_p = self.model(subframe2)
            loc_p = loc_p.detach().requires_grad_(False)
            conf_p = conf_p.detach().requires_grad_(False)
            loc_p,conf_p=self.encoder.scale_back_batch(loc_p,conf_p)
            conf_p = conf_p[:,:,1:] # ignore label 0 (background)

            # correct to actual image (relative size)
            loc_p.view(-1,4)[:,0]*=(fx/SSDSIZE)
            loc_p.view(-1,4)[:,2]*=(fx/SSDSIZE)
            loc_p.view(-1,4)[:,1]*=(fy/SSDSIZE)
            loc_p.view(-1,4)[:,3]*=(fy/SSDSIZE)
            loc_p.view(-1,4)[:,0]+=(x1/SSDSIZE)-2.0
            loc_p.view(-1,4)[:,2]+=(x1/SSDSIZE)-2.0
            loc_p.view(-1,4)[:,1]+=(y1/SSDSIZE)-2.0
            loc_p.view(-1,4)[:,3]+=(y1/SSDSIZE)-2.0

            # limit to bboxes IN the image
            nloc_p=loc_p.contiguous().squeeze(0)
            nconf_p=conf_p.contiguous().squeeze(0)
            keep=( (nloc_p[:,0]>0.01).logical_and(nloc_p[:,1]>0.01).logical_and(nloc_p[:,2]<0.99).logical_and(nloc_p[:,3]<0.99).logical_and((nloc_p[:,2]-nloc_p[:,0])>0.04).logical_and((nloc_p[:,3]-nloc_p[:,1])>0.04) ).nonzero().flatten()
            loc_p=nloc_p[keep,:].unsqueeze(0)
            conf_p=nconf_p[keep,].unsqueeze(0)

            loc_p.view(-1,4)[:,0]*= w
            loc_p.view(-1,4)[:,2]*= w
            loc_p.view(-1,4)[:,1]*= h
            loc_p.view(-1,4)[:,3]*= h

            locs.append(loc_p)
            confs.append(conf_p)

        bigframe=None
        smallframe=None
        tiles=self.alltiles(frame)
        aresults=[]
        for tile in tiles:
            loc_p,conf_p=self.dotile(frame,tile)
            #aresults.append(torch.tensor([[[18.0,1.0],[tile[0],tile[1]],[tile[0]+tile[2],tile[1]+tile[3]]]]))
            # AVOID BORDER rectangles
            nloc_p=loc_p.contiguous().squeeze(0)
            nconf_p=conf_p.contiguous().squeeze(0)
            keep=( (nloc_p[:,0]>0.01).logical_and(nloc_p[:,1]>0.01).logical_and(nloc_p[:,2]<0.99).logical_and(nloc_p[:,3]<0.99).logical_and((nloc_p[:,2]-nloc_p[:,0])>0.04).logical_and((nloc_p[:,3]-nloc_p[:,1])>0.04) ).nonzero().flatten()
            loc_p=nloc_p[keep,:].unsqueeze(0)
            conf_p=nconf_p[keep,].unsqueeze(0)


            loc_p.view(-1,4)[:,0]*= tile[2]
            loc_p.view(-1,4)[:,2]*= tile[2]
            loc_p.view(-1,4)[:,1]*= tile[3]
            loc_p.view(-1,4)[:,3]*= tile[3]
            loc_p.view(-1,4)[:,0]+= tile[0]
            loc_p.view(-1,4)[:,1]+= tile[1]
            loc_p.view(-1,4)[:,2]+= tile[0]
            loc_p.view(-1,4)[:,3]+= tile[1]
            locs.append(loc_p)
            confs.append(conf_p)
        conf_p=torch.cat(confs,1).contiguous()
        loc_p=torch.cat(locs,1).contiguous()

        #aresults=torch.cat(aresults,dim=0).unsqueeze(0)
        #logger.warn("we now have "+str(aresults.shape))
        #return aresults
        labels=conf_p # background is already removed earlier

        # and now we do nms
        labelmask = labels >= min_conf

        results=[]
        #nms needs to be done the slow way 
        for frame in range(labels.shape[0]):
            lboxes=[self.hardEmpty.view(0,3,2)]
            for label in range(labels.shape[2]):
                framelabelscores=labels[frame,:,label]
                frameboxes=loc_p[frame]
                indices=labelmask[frame,:,label].nonzero().flatten() #this is only needed because of buggy pytorch ONNX export, which doesn't like boolean mask access
                scores = torch.cat((framelabelscores[indices],self.hardZero.to(self.precision)),0)
                boxes = torch.cat((frameboxes[indices],self.hardZero.to(self.precision).view(1,1).expand(1,4)),0)

                # do non maximum suppression
                #if self.device.type=="cpu":
                # likely a laptop, we need to limit memory usage
                index = scores.argsort(descending=True)[0:5000] # only the 5000 highest scores per class are kept, speeds things up in a worst case scenario
                scores=scores[index].contiguous()
                boxes=boxes[index].contiguous()
                overlapping = (ssdutils.calc_iou_tensor(boxes,boxes) > nms_iou) # 2x2 boolean matrix of all boxes with too high jaccard overlap - each row has at least one True value on diagonal
                scorematrix = overlapping.float() * scores[:,None]       #this replaces the boolean values with the scores of the column-box

                keep = (scorematrix.max(dim=0)[0] == scores).nonzero().view(-1)

                # this is serialisable for export to ONNX
                scores = scores[keep].contiguous()
                boxes = boxes[keep].contiguous()

                #PROBLEM: with fp16 some overlapping boxes have the same score - non unique maximum
                #luckily there are not too manx so we just do this again and keep any random one
                overlapping = (ssdutils.calc_iou_tensor(boxes,boxes) > nms_iou)
                rands=torch.randn(scores.view(-1).shape,device=self.device,dtype=self.precision)
                randmatrix = overlapping.float() * rands[:,None]
                keep = (randmatrix.max(dim=0)[0] == rands).nonzero().view(-1)
                
                scores = scores[keep]
                boxes = boxes[keep].view(-1,2,2)

                score_enc = torch.cat( (scores.unsqueeze(1)*0 + label + 1, scores.unsqueeze(1)), 1).unsqueeze(1)
                boxes = torch.cat( (score_enc,boxes), 1)
                # boxes are now of shape [detections][3][2]
                # with content [[class,conf],[x1,y1],[x2,y2]], ...
                lboxes.append(boxes)
            lboxes = torch.cat( lboxes,0 )
            index = lboxes[:,0,1].sort(descending=True)[1]
            finalboxes=lboxes[index].contiguous()

            # make sure these are NOT in exclude
            coords=finalboxes[:,1:,:].view(-1,4)
            overlapping = ssdutils.calc_iou_tensor(exclude,coords) # len(cooords) x len(exclude) matrix
            bad = overlapping>nms_iou
            badexcludes=torch.max(bad,dim=1)[0]
            badcoords=torch.max(bad,dim=0)[0]
            index=badcoords.logical_not().nonzero().flatten()
            results.append(finalboxes[index].contiguous().cpu())

        return results

def yolo11_findnew(image, min_conf=0.25, nms_iou=0.45):
    """
    Simple YOLO11-based detector that mimics the output format of SSD.findnew
    for use by trackerAutoAnnotate.

    Returns:
        torch.Tensor of shape [N, 3, 2] with:
          [class_id, score], [x1, y1], [x2, y2]
        in image pixel coordinates.
    """
    global YOLO11NET, DEVICE

    if YOLO11NET is None:
        logger.warn("YOLO11NET is None - cannot run YOLO11 detection.")
        # Empty tensor with correct shape
        return torch.zeros((0, 3, 2), dtype=torch.float32)

    # YOLO expects RGB
    mimg_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    try:
        if hasattr(YOLO11NET, "predict"):
            results = YOLO11NET.predict(
                mimg_rgb,
                conf=min_conf,
                iou=nms_iou,
                verbose=False,
            )
        else:
            # Older ultralytics API: __call__ triggers predict
            results = YOLO11NET(mimg_rgb)
    except TypeError:
        # Some versions don't accept conf/iou in predict
        results = YOLO11NET.predict(mimg_rgb, verbose=False)

    if not results:
        return torch.zeros((0, 3, 2), dtype=torch.float32)

    res = results[0]
    boxes = getattr(res, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return torch.zeros((0, 3, 2), dtype=torch.float32)

    xyxy = boxes.xyxy.cpu()   # [N,4]
    scores = boxes.conf.cpu() # [N]
    cls = boxes.cls.cpu()     # [N]

    n = xyxy.shape[0]
    out = torch.zeros((n, 3, 2), dtype=torch.float32)
    # class id and score
    out[:, 0, 0] = cls
    out[:, 0, 1] = scores
    # coords
    out[:, 1, 0] = xyxy[:, 0]
    out[:, 1, 1] = xyxy[:, 1]
    out[:, 2, 0] = xyxy[:, 2]
    out[:, 2, 1] = xyxy[:, 3]

    return out
        
def trackerInit(config):
    global SSDMULTIBOX,DEVICE,RE3MODEL,CONFIG,CLASSIFIER,FLAGCLASS
    global YOLO11NET, YOLO11CLASS
    yolo_weights = config.get("yolo11model", None)
    
    #use a single instance of this, saves memory
    CONFIG=config
    if CONFIG['dnndevice'] is None:
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        DEVICE=torch.device(CONFIG['dnndevice'])
    SSDMULTIBOX=SSD()
    RE3MODEL = Re3Net().to(DEVICE)
    RE3MODEL.eval()
    if CONFIG["re3model"] is None:
        RE3PATH="checkpoint.pth"
        weights=torch.hub.load_state_dict_from_url('https://keeper.mpdl.mpg.de/f/55db32f79a464c9c8dd2/?dl=1', map_location=lambda storage, loc: storage, file_name=RE3PATH)
    else:
        weights=torch.load(CONFIG['re3model'], map_location=lambda storage, loc: storage)
    RE3MODEL.load_state_dict(weights)

    FLAGCLASS=[]
    if CONFIG["flagmodel"] is None:
        weights=None
        CLASSIFIER=None
    else:
        weights=torch.load(CONFIG['flagmodel'], map_location=lambda storage, loc: storage)
        CLASSIFIER=flagmodel.net(None,None,state_dict=weights).to(DEVICE)
        CLASSIFIER.eval()
        cs=CLASSIFIER.get_extra_state()
        for c in cs:
            FLAGCLASS.append(c)
            
    if UltralyticsYOLO is not None and yolo_weights is not None:
        try:
            logger.info("Initialising YOLO11 detector with weights '{}'".format(yolo_weights))
            YOLO11NET = UltralyticsYOLO(yolo_weights)
            # put model on same device as the other networks if possible
            try:
                YOLO11NET.to(str(DEVICE))
            except Exception:
                pass

            # try to get class names
            names = None
            if hasattr(YOLO11NET, "names"):
                names = YOLO11NET.names
            elif hasattr(YOLO11NET, "model") and hasattr(YOLO11NET.model, "names"):
                names = YOLO11NET.model.names
            YOLO11CLASS = names
        except Exception as ex:
            logger.warn("Failed to initialise YOLO11 detector: {}. Using SSD only.".format(ex))
            YOLO11NET = None
            YOLO11CLASS = None
    elif yolo_weights is not None and UltralyticsYOLO is None:
        logger.warn("yolo11model specified but ultralytics is not installed. Using SSD only.")            

def yolo11_correct_box(image, pbox, ref_bbox):
    """
    Use YOLO11 to refine the tracker box.

    Args:
        image: current BGR frame (H,W,3) as np.ndarray
        pbox:  [x1, y1, x2, y2] from Re3 (float or int)
        ref_bbox: reference bbox (e.g. from previous annotation) [x1,y1,x2,y2]

    Returns:
        list [x1,y1,x2,y2] if YOLO11 found a good match, otherwise None.
    """
    global YOLO11NET

    if YOLO11NET is None:
        return None

    pbox = np.array(pbox, dtype=float)
    ref_bbox = np.array(ref_bbox, dtype=float)

    H, W = image.shape[0], image.shape[1]

    # Compute crop around current predicted box (same logic as SSD.prepare)
    w = pbox[2] - pbox[0]
    h = pbox[3] - pbox[1]
    x0 = pbox[0] + w / 2.0
    y0 = pbox[1] + h / 2.0
    d = h
    if w > h:
        d = w

    crop_factor = float(CONFIG.get("ssd_track_crop_factor", 3.0))
    x1 = x0 - d * crop_factor * 0.5
    x2 = x0 + d * crop_factor * 0.5
    y1 = y0 - d * crop_factor * 0.5
    y2 = y0 + d * crop_factor * 0.5

    x1, y1, x2, y2 = [int(a) if a >= 0 else 0 for a in [x1, y1, x2, y2]]
    x1, x2 = [a if a <= W - 1 else W - 1 for a in [x1, x2]]
    y1, y2 = [a if a <= H - 1 else H - 1 for a in [y1, y2]]

    if x2 <= x1 or y2 <= y1:
        return None

    roi = image[y1:y2, x1:x2, :]
    if roi.size == 0:
        return None

    # YOLO expects RGB
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    conf_thr = float(CONFIG.get("yolo11_conf_threshold", 0.25))
    iou_nms = float(CONFIG.get("yolo11_iou_threshold", 0.45))

    try:
        if hasattr(YOLO11NET, "predict"):
            results = YOLO11NET.predict(
                roi_rgb,
                conf=conf_thr,
                iou=iou_nms,
                verbose=False,
            )
        else:
            results = YOLO11NET(roi_rgb)
    except TypeError:
        results = YOLO11NET.predict(roi_rgb, verbose=False)

    if not results:
        return None

    res = results[0]
    boxes = getattr(res, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None

    # Convert to a normal NumPy array (detaches from inference-mode tensor)
    xyxy = boxes.xyxy.detach().cpu().numpy()  # [N,4]

    # Map boxes back to full-image coordinates
    xyxy[:, 0] += x1
    xyxy[:, 1] += y1
    xyxy[:, 2] += x1
    xyxy[:, 3] += y1

    # Choose box with highest IoU vs reference bbox
    candidates = torch.tensor(xyxy, dtype=torch.float32)  # [N,4]
    ref = torch.tensor(ref_bbox.reshape(1, 4), dtype=torch.float32)
    ious = ssdutils.calc_iou_tensor(ref, candidates).view(-1)
    if ious.numel() == 0:
        return None

    best_idx = int(torch.argmax(ious))
    best_iou = float(ious[best_idx])

    min_iou = float(CONFIG.get("ssd_re3_correction_min_iou", 0.8))
    if best_iou < min_iou:
        logger.info("YOLO11 correction below IoU threshold: %.3f" % best_iou)
        return None

    best_box = candidates[best_idx].tolist()
    logger.info("YOLO11 correction box %s with IoU %.3f" % (str(best_box), best_iou))
    return best_box



class Re3Tracker(object):
    def __init__(self, model_path = 'checkpoint.pth'):
        self.net=RE3MODEL
        self.device=DEVICE
        self.tracked_data = {}

    def track(self, id, image, prev_image=None, bbox = None, past_bbox=None):

        if bbox is not None:
            # trick - only reinit bounding box, not the whole state
            if id in self.tracked_data:
                lstm_state, initial_state, forward_count = self.tracked_data[id]
            else:
                lstm_state = None
                forward_count = 0
            prev_image = image
            past_bbox = copy.deepcopy(bbox)
        elif id in self.tracked_data:
            lstm_state, initial_state, forward_count = self.tracked_data[id]
        else:
            raise Exception('Id {0} without initial bounding box'.format(id))

        if (bbox is not None) or (TRANSROTATION is None):
            corrected_bbox=past_bbox
        else:
            corrected_bbox=past_bbox.reshape(2,2).astype(float)
            corrected_bbox=int(CONFIG['global_translation_track_size'])*(corrected_bbox-TRANSROTATION[1])/TRANSROTATION[2]
            corrected_bbox=np.concatenate((corrected_bbox,np.array([[1.],[1.]])),axis=1)
            corrected_bbox=np.concatenate([ [np.dot(TRANSROTATION[0],point)] for point in corrected_bbox])


            corrected_bbox=(corrected_bbox[:,[[0,1]]]*TRANSROTATION[2]/int(CONFIG['global_translation_track_size']))+TRANSROTATION[1]
            corrected_bbox=corrected_bbox.reshape(4).astype(int)

        cropped_input0, past_bbox_padded = im_util.get_cropped_input(prev_image, past_bbox, CROP_PAD, CROP_SIZE)
        cropped_input1, new_bbox_padded = im_util.get_cropped_input(image, corrected_bbox, CROP_PAD, CROP_SIZE)

        network_input = np.stack((cropped_input0.transpose(2,0,1),cropped_input1.transpose(2,0,1)))
        network_input = torch.tensor(network_input, dtype = torch.float)[:,[2,0,1],:,:].contiguous()
        
        with torch.no_grad():
            network_input = network_input.to(self.device)
            network_predicted_bbox, lstm_state = self.net(network_input, prevLstmState = lstm_state)

        if forward_count == 0:
            initial_state = lstm_state
            # initial_state = None
            
        predicted_bbox = bb_util.from_crop_coordinate_system(network_predicted_bbox.cpu().data.numpy()/10, new_bbox_padded, 1, 1)

        # Reset state
        if forward_count > 0 and forward_count % MAX_TRACK_LENGTH == 0:
            lstm_state = initial_state

        forward_count += 1

        if bbox is not None:
            predicted_bbox = bbox

        predicted_bbox = predicted_bbox.reshape(4)

        self.tracked_data[id] = (lstm_state, initial_state, forward_count)
        
        return predicted_bbox

    def reset(self):
        self.tracked_data = {}

def getRectForTracker(img, shape):
    H = img.shape[0]
    W = img.shape[1]

    qrect = shape.boundingRect()
    tl = qrect.topLeft()
    h = qrect.height()
    w = qrect.width()

    x = tl.x()
    y = tl.y()
    x2 = x+w
    y2 = y+h
    x,y,x2,y2 = [ (a if a>=0 else 0) for a in [x,y,x2,y2]]
    x,x2 = [ (a if a<=W-1 else W-1) for a in [x,x2]]
    y,y2 = [ (a if a<=H-1 else H-1) for a in [y,y2]]

    x= x if x>=0 else 0
    y= y if y>=0 else 0

    return [int(_) for _ in [x,y,x2,y2]]

class Tracker():

    def __init__(self, *args, **kwargs):
        self.tracker=Re3Tracker()
        self.shape = None
        self.newshape = None

    @property
    def isRunning(self):
        return (self.shape is not None)


    def initTracker(self, shape):
        status = False
        if REFIMAGE is None or not shape:
            logger.warn("invalid tracker initialisation")
            return status
        else:
            self.shape=shape
            try:
               if shape.flags['disable_visual_tracking']:
                   return True
            except:
                pass
            fimg = REFIMAGE
            srect = getRectForTracker(fimg, shape)
            rrect = None
            if (self.newshape is not None):
                rrect = getRectForTracker(fimg, self.newshape)
            logger.info("rectangles:"+str((srect,rrect)))
            if ((rrect is None) or rrect[0]!=srect[0] or rrect[1]!=srect[1] or rrect[2]!=srect[2] or rrect[3]!=srect[3]):
                logger.info("re-initializing tracker")
                _ = self.tracker.track(1,fimg,bbox=np.array([srect[0],srect[1],srect[2],srect[3]]))
            else:
                logger.info("keeping tracker")
            self.newshape=None

            status = True

        return status

    def updateTracker(self, shape):

        assert (shape and shape.label == self.shape.label),"Invalid tracker state!"

        status = False

        result = shape
        try:
            if shape.flags['disable_visual_tracking']:
                return result, True
        except:
            pass

        if not self.isRunning or CURIMAGE is None:
            if not self.isRunning:
                logger.warning("tracker is not running")
            else:
                logger.warning("no image")
            return result, status

        #mimg = CURIMAGE
        #srect = getRectForTracker(mimg,self.shape)
        #pbox = self.tracker.track(1,mimg,prev_image=REFIMAGE,past_bbox=np.array([srect[0],srect[1],srect[2],srect[3]]))
        #logger.info("tracker reported:"+str(pbox)+" running SSD")
        #prep = SSDMULTIBOX.prepare(mimg,pbox)
        #ssd1box = SSDMULTIBOX.detect(prep,pbox)
        #ssd2box = SSDMULTIBOX.detect(prep,srect)
        #if ssd1box is None:
            #ssdbox=ssd2box
        #elif ssd2box is None:
            #ssdbox=ssd1box
        #else:
            #logger.info("combining ssd boxes")
            #ssdbox=list(float(CONFIG['ssd_re3_correction_alpha'])*np.array(ssd1box)+(1.0-float(CONFIG['ssd_re3_correction_alpha']))*np.array(ssd2box))
        #if ssdbox is not None:
            #logger.info("merging with Re3")
            #pbox=list(float(CONFIG['ssd_re3_correction_alpha'])*np.array(ssdbox)+(1.0-float(CONFIG['ssd_re3_correction_alpha']))*np.array(pbox))
            #logger.info("ssd fusion reported:"+str(pbox))

        #logger.info("fixed shape:"+str(pbox))
        
        mimg = CURIMAGE
        srect = getRectForTracker(mimg, self.shape)
        pbox = self.tracker.track(
            1,
            mimg,
            prev_image=REFIMAGE,
            past_bbox=np.array([srect[0], srect[1], srect[2], srect[3]]),
        )

        logger.info("tracker reported: %s" % str(pbox))

        # Decide which correction backend to use: YOLO11 (if available) or SSD (legacy)
        use_yolo = (YOLO11NET is not None) and (CONFIG.get("yolo11model", None) is not None)

        ssdbox = None
        if use_yolo:
            logger.info("running YOLO11 correction")
            ssdbox = yolo11_correct_box(mimg, pbox, srect)
        else:
            logger.info("running SSD correction")
            prep = SSDMULTIBOX.prepare(mimg, pbox)
            ssd1box = SSDMULTIBOX.detect(prep, pbox)
            ssd2box = SSDMULTIBOX.detect(prep, srect)
            if ssd1box is None:
                ssdbox = ssd2box
            elif ssd2box is None:
                ssdbox = ssd1box
            else:
                logger.info("combining ssd boxes")
                alpha = float(CONFIG['ssd_re3_correction_alpha'])
                ssdbox = list(
                    alpha * np.array(ssd1box)
                    + (1.0 - alpha) * np.array(ssd2box)
                )

        if ssdbox is not None:
            logger.info("merging detection with Re3")
            alpha = float(CONFIG['ssd_re3_correction_alpha'])
            pbox = list(
                alpha * np.array(ssdbox)
                + (1.0 - alpha) * np.array(pbox)
            )
            logger.info("fusion reported: %s" % str(pbox))

        logger.info("fixed shape:" + str(pbox))
       
        fw=float(pbox[2]-pbox[0])/float(srect[2]-srect[0])
        fh=float(pbox[3]-pbox[1])/float(srect[3]-srect[1])
        cx=pbox[0]-(srect[0]*fw)
        cy=pbox[1]-(srect[1]*fh)

        warp_matrix = np.eye(2, 3, dtype=np.float32)
        warp_matrix[0,0]=fw
        warp_matrix[1,1]=fh
        warp_matrix[0,2]=cx
        warp_matrix[1,2]=cy
        logger.info("warp matrix:"+str(warp_matrix))

        cc = False

        if (fw==fw and fh==fh):
            result = copy.deepcopy(shape)
            result.transform(warp_matrix)
            trect = getRectForTracker(mimg, result)
            logger.info("resulting rect:"+str(trect))
            status = True
            self.newshape=copy.deepcopy(result)
            logger.info("Tracker succeeded")
        else:
            logger.warning("Tracker failed")

        return result , status

    def __reset__(self):
        self.shape = None
        self.tracker = None

    def stopTracker(self):
        self.__reset__()

def crop(image,dimensions):
    x,y,w,h=dimensions
    if w>h:
        cw=int(w/2)
    else:
        cw=int(h/2)
    l=(x-int(w/2))+cw
    r=l+w
    t=(y-int(h/2))+cw
    b=t+h
    p=torchvision.transforms.Pad(cw,padding_mode='reflect')
    pimage=p(image)
    return pimage[:,t:b,l:r]

def scale(image,dimensions):
    w,h=dimensions
    t=torchvision.transforms.Resize((h,w),torchvision.transforms.InterpolationMode.BILINEAR,antialias=True)
    return t(image)

def trackerFindGlobalOffset(aqimg,qimg):
    global REFIMAGE,CURIMAGE,TRANSROTATION

    REFIMAGE = ocvutil.qtImg2CvMat(aqimg)
    CURIMAGE = ocvutil.qtImg2CvMat(qimg)

    if (int(CONFIG['global_translation_track_size'])<=0):
        TRANSROTATION=None
        return
    w=REFIMAGE.shape[1]
    h=REFIMAGE.shape[0]
    size=min(h,w)
    center=(np.array([w,h],dtype=int)//2)
    lt=center-(size//2)
    frame0r = cv2.cvtColor(cv2.resize(REFIMAGE[lt[1]:lt[1]+size,lt[0]:lt[0]+size], (int(CONFIG['global_translation_track_size']),int(CONFIG['global_translation_track_size']))),cv2.COLOR_BGR2GRAY)
    frame1r = cv2.cvtColor(cv2.resize(CURIMAGE[lt[1]:lt[1]+size,lt[0]:lt[0]+size], (int(CONFIG['global_translation_track_size']),int(CONFIG['global_translation_track_size']))),cv2.COLOR_BGR2GRAY)
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    TRANSROTATION=None

    try:
        (_, T) = cv2.findTransformECC (frame0r,frame1r,warp_matrix,cv2.MOTION_AFFINE, \
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000,1e-10), None, 1)
        TRANSROTATION=(T,lt,size)
    except:
        TRANSROTATION=None


def trackerDetectFlags(qimg,shapes):
    if CLASSIFIER is None:
        return
    mimg = ocvutil.qtImg2CvMat(qimg)
    frame=torch.from_numpy(mimg).to(DEVICE).permute(2,0,1)[[2,1,0],:,:] * (1.0/255.0)
    w=frame.shape[2]
    h=frame.shape[1]
    for shape in shapes:
        if shape.flags['auto_flag']==True:
            rect=getRectForTracker(mimg,shape)
            x=int((rect[0]+rect[2])/2)
            y=int((rect[1]+rect[3])/2)
            w=rect[2]-rect[0]
            h=rect[3]-rect[1]
            if (w>h):
                h=w
            else:
                w=h
            w=int(w*1.42)
            cframe=scale(crop(frame,[x,y,w,w]),[SSDSIZE,SSDSIZE])
            result=CLASSIFIER(cframe.unsqueeze(0)).squeeze(0)
            c=result.argmax().item()
            for flag in FLAGCLASS:
                if flag in shape.flags:
                    shape.flags[flag]=False
            if c:
                shape.flags[FLAGCLASS[c]]=True

#def trackerAutoAnnotate(qimg,shapes):
    #mimg = ocvutil.qtImg2CvMat(qimg)
    #rects=[]
    #for shape in shapes:
        #rects.append(getRectForTracker(mimg,shape))
    #newrects=SSDMULTIBOX.findnew(mimg,rects)[-1]
    #newrects=newrects[newrects[:,0,1]>float(CONFIG['ssd_threshold_autoannotation'])]
    #id0=int(random.random()*100000)*1000
    #id1=0
    #newshapes=[]
    #for rect in newrects:
        #myid=id0+id1
        #myname=COCOCLASS[int(rect[0,0])]+("_%d"%(myid))
        #shape=Shape(label=myname,shape_type="rectangle",flags={})
        #shape.insertPoint(1,QtCore.QPoint(int(rect[1,0]),int(rect[1,1])))
        #shape.insertPoint(2,QtCore.QPoint(int(rect[2,0]),int(rect[2,1])))
        #id1+=1
        #shape.flags['auto_flag']=True
        #newshapes.append(shape)
    #trackerDetectFlags(qimg,newshapes) # modifies newshapes
    #return newshapes
    
def trackerAutoAnnotate(qimg, shapes):
    global YOLO11NET, YOLO11CLASS

    mimg = ocvutil.qtImg2CvMat(qimg)
    rects = []
    for shape in shapes:
        rects.append(getRectForTracker(mimg, shape))

    use_yolo = YOLO11NET is not None and CONFIG.get("yolo11model", None) is not None

    if use_yolo:
        # Use YOLO11 (ultralytics) for auto-annotation
        conf = float(CONFIG.get("yolo11_conf_threshold", 0.25))
        iou = float(CONFIG.get("yolo11_iou_threshold", 0.45))
        newrects = yolo11_findnew(mimg, min_conf=conf, nms_iou=iou)
    else:
        # Original SSD-based auto-annotation
        newrects = SSDMULTIBOX.findnew(mimg, rects)[-1]
        newrects = newrects[
            newrects[:, 0, 1] > float(CONFIG['ssd_threshold_autoannotation'])
        ]

    # Nothing found
    if newrects.shape[0] == 0:
        return []

    id0 = int(random.random() * 100000) * 1000
    id1 = 0
    newshapes = []

    # Helper to get class name
    def class_name_from_id(idx):
        idx = int(idx)
        if use_yolo and YOLO11CLASS is not None:
            # ultralytics names can be list or dict
            if isinstance(YOLO11CLASS, dict):
                return YOLO11CLASS.get(idx, str(idx))
            elif isinstance(YOLO11CLASS, (list, tuple)):
                if 0 <= idx < len(YOLO11CLASS):
                    return YOLO11CLASS[idx]
            return str(idx)
        else:
            # SSD / COCO pathway (existing behaviour)
            if 0 <= idx < len(COCOCLASS):
                return COCOCLASS[idx]
            return str(idx)

    for rect in newrects:
        myid = id0 + id1
        cname = class_name_from_id(rect[0, 0])
        myname = cname + ("_%d" % (myid))
        shape = Shape(label=myname, shape_type="rectangle", flags={})
        shape.insertPoint(1, QtCore.QPoint(int(rect[1, 0]), int(rect[1, 1])))
        shape.insertPoint(2, QtCore.QPoint(int(rect[2, 0]), int(rect[2, 1])))
        id1 += 1
        shape.flags['auto_flag'] = True
        newshapes.append(shape)

    trackerDetectFlags(qimg, newshapes)  # modifies newshapes
    return newshapes
    
