import visdom
import time
import numpy as np
import string

class Visualizer(object):
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = {}

    def plot_many_stack(self, d):
        '''
        self.plot('loss',1.00)
        '''
        name = list(d.keys())
        name_total = " ".join(name)
        x = self.index.get(name_total, 0)
        val = list(d.values())
        if len(val) == 1:
            y = np.array(val)
        else:
            y = np.array(val).reshape(-1, len(val))
        # print(x)
        self.vis.line(Y=y, X=np.ones(y.shape) * x,
                      win=str(name_total),  # unicode
                      opts=dict(legend=name,title=name_total),
                      update=None if x == 0 else 'append')
        self.index[name_total] = x + 1
    def plot_many_list(self,value,name):
        if isinstance(name,str) == True:
            name = list(name)
        else:
            c = 0
            name_total = " ".join(name)
            x = self.index.get(name_total, 0)
if __name__ == '__main__':
    vis = Visualizer(env = 'show')
    for i in range(10):
        y = [m for m in range (i)]
        for va in y:
            vis.plot_many_stack({'train_loss': va,'test_loss':va + 1})
