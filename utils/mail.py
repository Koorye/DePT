import yagmail


class MailClient(object):
    def __init__(self, cfg):
        self.client = yagmail.SMTP(cfg['username'], cfg['password'], cfg['host'])
        self.to = cfg['to']
    
    def send(self, subject, contents):
        self.client.send(self.to, subject, contents)


if __name__ == '__main__':
    cfg = dict(
        username='1311586225@qq.com',
        password='ftzcotjfnflwbabb',
        host='smtp.qq.com',
        to='1311586225@qq.com',
    )
    client = MailClient(cfg)
    client.send('test', 'this is a test content')
