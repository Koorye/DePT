# a simple mail sending tool

import yagmail


class MailClient(object):
    def __init__(self, cfg):
        self.client = yagmail.SMTP(cfg['username'], cfg['password'], cfg['host'])
        self.to = cfg['to']
    
    def send(self, subject, contents):
        self.client.send(self.to, subject, contents)
