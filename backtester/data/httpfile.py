"""Seekable read-only file over HTTP range requests, so pyarrow can read a remote
parquet footer and only the row groups / columns it needs."""
import io

import requests


class HTTPFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (for pyarrow footers/row groups)."""
    def __init__(self,url):
        self.url=url; self.s=requests.Session()
        h=self.s.head(url,allow_redirects=True,timeout=60); self.url=h.url; self.size=int(h.headers['Content-Length']); self.pos=0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self,off,whence=0):
        self.pos={0:off,1:self.pos+off,2:self.size+off}[whence]; return self.pos
    def read(self,n=-1):
        if n is None or n<0: n=self.size-self.pos
        if n==0 or self.pos>=self.size: return b''
        end=min(self.pos+n,self.size)-1
        for k in range(4):
            try:
                r=self.s.get(self.url,headers={'Range':f'bytes={self.pos}-{end}'},timeout=120); r.raise_for_status(); break
            except Exception:
                if k==3: raise
        self.pos+=len(r.content); return r.content
    def readinto(self,b):
        d=self.read(len(b)); b[:len(d)]=d; return len(d)
