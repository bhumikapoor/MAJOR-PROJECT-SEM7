from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import base64, io, json, time, traceback
import numpy as np
from PIL import Image, ImageDraw
from sklearn.ensemble import RandomForestClassifier
from threading import Lock

MODEL = None
MODEL_INFO = None
MODEL_LOCK = Lock()

ROOT = Path(__file__).parent
MAX_BYTES = 20 * 1024 * 1024

def data_url(im):
    out = io.BytesIO(); im.save(out, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(out.getvalue()).decode()
def decode(s):
    if not isinstance(s, str) or ',' not in s: raise ValueError('Missing image data')
    raw = base64.b64decode(s.split(',', 1)[1], validate=True)
    if len(raw) > MAX_BYTES: raise ValueError('Image exceeds 20 MB')
    im = Image.open(io.BytesIO(raw)); im.load()
    if im.width * im.height > 16000000: raise ValueError('Image exceeds 16 megapixels')
    return im
def binary(im):
    a = np.array(im.convert('L'))
    return a >= 128

def demo():
    rng = np.random.default_rng(8); h,w=480,720
    a=np.zeros((h,w,3),dtype=np.uint8); noise=rng.normal(0,9,(h,w))
    for i,c in enumerate((99,105,65)): a[:,:,i]=np.clip(c+noise,0,255)
    im=Image.fromarray(a); d=ImageDraw.Draw(im); mask=Image.new('L',(w,h)); md=ImageDraw.Draw(mask)
    for y in range(28,h,48):
        for x in range(25,w,34):
            dx=int(rng.integers(-8,9)); dy=int(rng.integers(-7,8)); color=(int(rng.integers(47,80)),int(rng.integers(103,147)),int(rng.integers(45,75)))
            d.ellipse((x+dx-12,y+dy-6,x+dx+12,y+dy+6),fill=color)
    for x,y,rx,ry in [(212,165,54,30),(455,282,68,39),(562,112,32,23),(170,348,27,17)]:
        for j in range(90):
            dx=int(rng.normal(0,rx/2)); dy=int(rng.normal(0,ry/2))
            if (dx/rx)**2+(dy/ry)**2<1:
                d.ellipse((x+dx-5,y+dy-4,x+dx+5,y+dy+4), fill=(61,132,61)); md.ellipse((x+dx-5,y+dy-4,x+dx+5,y+dy+4),fill=255)
    return {'image':data_url(im),'reference':data_url(mask)}

def features(rgb):
    a = rgb.astype(np.float32) / 255.0
    r,g,b = a[:,:,0],a[:,:,1],a[:,:,2]
    total = np.maximum(r+g+b, 1e-5)
    return np.stack([r,g,b,2*g-r-b,(g-r)/(g+r+1e-5),r/total,g/total,b/total], axis=-1).reshape(-1,8)

def train(payload):
    global MODEL, MODEL_INFO
    image=decode(payload['image']).convert('RGB'); ref=decode(payload.get('reference',''))
    if image.size != ref.size: raise ValueError('Training image and reference mask dimensions must match')
    y=binary(ref).reshape(-1).astype(np.uint8)
    if y.min()==y.max(): raise ValueError('Training mask needs both foreground and background pixels')
    rng=np.random.default_rng(42); indexes=[]
    for label in (0,1):
        choices=np.flatnonzero(y==label); indexes.append(rng.choice(choices, min(len(choices),15000),replace=False))
    indexes=np.concatenate(indexes); X=features(np.asarray(image))[indexes]
    model=RandomForestClassifier(n_estimators=60,max_depth=18,min_samples_leaf=3,n_jobs=-1,random_state=42)
    model.fit(X,y[indexes])
    info={'training_pixels':len(indexes),'foreground_samples':int((y[indexes]==1).sum()),'background_samples':int((y[indexes]==0).sum()),'image_width':image.width,'image_height':image.height}
    with MODEL_LOCK: MODEL=model; MODEL_INFO=info
    return info

def analyze(payload):
    start=time.perf_counter(); image=decode(payload['image']).convert('RGB'); rgb=np.asarray(image,dtype=np.int16); h,w=rgb.shape[:2]
    mode=payload.get('mode','heuristic'); threshold=int(payload.get('threshold',20))
    if not -100 <= threshold <= 255: raise ValueError('Threshold must be between -100 and 255')
    if mode=='mask':
        predim=decode(payload.get('prediction',''))
        if predim.size != image.size: raise ValueError('Prediction mask dimensions must match the image')
        pred=binary(predim); method='Uploaded prediction mask (values ≥ 128)'
    elif mode=='heuristic':
        pred=(2*rgb[:,:,1]-rgb[:,:,0]-rgb[:,:,2]) >= threshold; method='RGB excess-green heuristic (2G − R − B)'
    elif mode=='learned':
        with MODEL_LOCK: model=MODEL; info=MODEL_INFO
        if model is None: raise ValueError('Train the pixel classifier first using an image and reference mask')
        X=features(rgb); pieces=[]
        for start_idx in range(0,len(X),100000): pieces.append(model.predict(X[start_idx:start_idx+100000]))
        pred=np.concatenate(pieces).reshape(h,w).astype(bool); method='User-trained random forest pixel classifier'
    else: raise ValueError('Unknown analysis mode')
    ref=None
    if payload.get('reference'):
        refim=decode(payload['reference'])
        if refim.size != image.size: raise ValueError('Reference mask dimensions must match the image')
        ref=binary(refim)
    n=int(pred.sum()); total=int(pred.size)
    overlay=rgb.astype(np.uint8).copy(); overlay[pred]=(overlay[pred]*0.48 + np.array([255,177,54])*0.52).astype(np.uint8)
    mask=Image.fromarray((pred*255).astype(np.uint8)); result={'width':w,'height':h,'positive_pixels':n,'total_pixels':total,'coverage':n/total,'overlay':data_url(Image.fromarray(overlay)),'mask':data_url(mask),'method':method,'threshold':threshold if mode=='heuristic' else (128 if mode=='mask' else None),'elapsed_ms':round((time.perf_counter()-start)*1000,1),'metrics':None}
    grid_size=int(payload.get('grid_size', 8))
    if grid_size not in (4,8,12): raise ValueError('Grid size must be 4, 8, or 12')
    # Integer boundaries avoid losing pixels when dimensions are not divisible by grid size.
    grid=[]; grid_img=np.asarray(image).copy().astype(np.uint8)
    for row in range(grid_size):
        for col in range(grid_size):
            y0=row*h//grid_size; y1=(row+1)*h//grid_size; x0=col*w//grid_size; x1=(col+1)*w//grid_size
            cell=pred[y0:y1,x0:x1]; count=int(cell.sum()); valid=int(cell.size)
            cover=count/valid if valid else 0
            grid.append({'row':row+1,'column':col+1,'x0':x0,'y0':y0,'width':x1-x0,'height':y1-y0,'positive_pixels':count,'total_pixels':valid,'coverage':cover})
            if valid:
                color=np.array([235,95,50]) if cover>=0.5 else (np.array([242,177,63]) if cover>=0.2 else np.array([91,154,104]))
                grid_img[y0:y1,x0:x1]=(grid_img[y0:y1,x0:x1]*0.45+color*0.55).astype(np.uint8)
                grid_img[y0:y0+1,x0:x1]=[245,245,235]; grid_img[y0:y1,x0:x0+1]=[245,245,235]
    result['grid_size']=grid_size; result['grid']=grid; result['grid_image']=data_url(Image.fromarray(grid_img))
    if mode=='learned': result['model_info']=info
    if ref is not None:
        tp=int((pred & ref).sum()); fp=int((pred & ~ref).sum()); fn=int((~pred & ref).sum()); tn=int((~pred & ~ref).sum())
        def div(a,b): return a/b if b else None
        result['metrics']={'tp':tp,'fp':fp,'fn':fn,'tn':tn,'iou':div(tp,tp+fp+fn),'dice':div(2*tp,2*tp+fp+fn),'precision':div(tp,tp+fp),'recall':div(tp,tp+fn)}
        errors=np.zeros((h,w,3),dtype=np.uint8); errors[:]=rgb.astype(np.uint8)*0.35; errors[pred & ref]=[78,191,146]; errors[pred & ~ref]=[244,157,65]; errors[~pred & ref]=[220,92,105]; result['errors']=data_url(Image.fromarray(errors))
    return result

class Handler(BaseHTTPRequestHandler):
    def send(self,status,content,kind):
        self.send_response(status); self.send_header('Content-Type',kind); self.send_header('Content-Length',str(len(content))); self.send_header('Cache-Control','no-store'); self.end_headers(); self.wfile.write(content)
    def do_GET(self):
        if self.path=='/api/demo': return self.send(200,json.dumps(demo()).encode(),'application/json')
        files={'/':'index.html','/app.js':'app.js','/style.css':'style.css'}
        if self.path not in files: return self.send(404,b'Not found','text/plain')
        p=ROOT/files[self.path]; return self.send(200,p.read_bytes(),{'html':'text/html; charset=utf-8','js':'text/javascript; charset=utf-8','css':'text/css; charset=utf-8'}[p.suffix[1:]])
    def do_POST(self):
        if self.path not in ('/api/analyze','/api/train'): return self.send(404,b'Not found','text/plain')
        try:
            size=int(self.headers.get('Content-Length','0'))
            if size<=0 or size>85000000: raise ValueError('Request too large or empty')
            payload=json.loads(self.rfile.read(size)); result=train(payload) if self.path=='/api/train' else analyze(payload); self.send(200,json.dumps(result).encode(),'application/json')
        except Exception as e: self.send(400,json.dumps({'error':str(e)}).encode(),'application/json')
if __name__=='__main__':
    import webbrowser, threading
    server=ThreadingHTTPServer(('127.0.0.1',8765),Handler)
    print('Fieldlens running at http://127.0.0.1:8765', flush=True)
    threading.Timer(0.8, lambda:webbrowser.open('http://127.0.0.1:8765')).start()
    try: server.serve_forever()
    except KeyboardInterrupt: pass
