import copy


base = dict(
    # dataset configs
    data = dict(
        root='root of datasets here',
        datasets_base_to_new=['dtd', 'caltech101', 'eurosat', 'ucf101', 'oxford_flowers', 
                              'oxford_pets', 'stanford_cars', 'fgvc_aircraft', 'food101', 'sun397', 'imagenet'],
        datasets_cross_dataset=['caltech101', 'oxford_pets', 'stanford_cars', 'oxford_flowers', 'food101',
                                'fgvc_aircraft', 'sun397', 'dtd', 'eurosat', 'ucf101',
                                'imagenetv2', 'imagenet_sketch', 'imagenet_a', 'imagenet_r'],
    ),

    # mail configs
    mail = dict(
        username='somebody@example.com',
        password='password here',
        host='host here',
        to='somebody@example.com',
    ),
)

##########################################################

coop = dict(
    # GPU ids, if you have multiple GPUs, it can be setted to [0, 1, 2, ...]
    # number of GPU ids is recommanded to be a multiple of 3
    # because seeds are 1, 2, 3
    gpu_ids = [0],
    # gpu_ids = [0, 1, 2],
    # training and eval mode
    # 'b2n' means base to new, or 'xd' means cross dataset and domain generalization
    mode='b2n',
    
    # training configs
    train = dict(
        trainer='CoOp',              # trainer, please see trainers
        cfg='vit_b16_ep10_bs4_lr35', # config, please see configs/
        seeds=[1, 2, 3],             # seeds
        loadep=-1,                   # load epoch, -1 to load the last epoch
        shots=16,                    # num of shots
        opts=[],                     # extra opts, if you have, please add, such as [OPTIM.MAX_EPOCH, 10]
    ),
    
    # grid search configs, if enable=False, grid search will not be used
    grid_search = dict(enable=False),
    
    # output configs
    output = dict(
        root='outputs/coop',   # output root
        result='results/coop', # result root 
        remove_dirs=['root'],  # which directorys will be removed before training task starts
    ),
)

cocoop = dict(
    gpu_ids = [0],
    mode='b2n',
    
    train = dict(
        trainer='CoCoOp',              
        cfg='vit_b16_c4_ep10_batch1_ctxv1', 
        seeds=[1, 2, 3],             
        loadep=-1,                   
        shots=16,                   
        opts=[],      
    ),
    
    grid_search = dict(enable=False),
    
    output = dict(
        root='outputs/cocoop',  
        result='results/cocoop',
        remove_dirs=['root'],       
    ),
)

kgcoop = dict(
    gpu_ids = [0],
    mode='b2n',
    
    train = dict(
        trainer='KgCoOp',              
        cfg='vit_b16_ep10_ctxv1_bs4_lr35', 
        seeds=[1, 2, 3],             
        loadep=-1,                   
        shots=16,                   
        opts=[],      
    ),
    
    grid_search = dict(enable=False),
    
    output = dict(
        root='outputs/kgcoop',  
        result='results/kgcoop',
        remove_dirs=['root'],       
    ),
)

maple = dict(
    gpu_ids = [0],
    mode='b2n',
    
    train = dict(
        trainer='MaPLe',              
        cfg='vit_b16_c2_ep10_batch4_2ctx', 
        seeds=[1, 2, 3],             
        loadep=-1,                   
        shots=16,                   
        opts=[],      
    ),
    
    grid_search = dict(enable=False),
    
    output = dict(
        root='outputs/maple',  
        result='results/maple',
        remove_dirs=['root'],       
    ),
)

coop_dept = dict(
    gpu_ids = [0],
    mode='b2n',
    
    train = dict(
        trainer='ExtrasLinearProbeCoOp',      
        cfg='vit_b16_ep10_bs4_lr35',
        seeds=[1, 2, 3],    
        loadep=-1,         
        shots=16,   
        opts=[],    
    ),
    
    grid_search = dict(enable=False),
    
    output = dict(
        root='outputs/coop_dept',   
        result='results/coop_dept', 
        remove_dirs=['root'],  
    ),
)

cocoop_dept = dict(
    gpu_ids = [0],
    mode='b2n',
    
    train = dict(
        trainer='ExtrasLinearProbeCoCoOp',              
        cfg='vit_b16_c4_ep10_batch1_ctxv1', 
        seeds=[1, 2, 3],             
        loadep=-1,                   
        shots=16,                   
        opts=[],      
    ),
    
    grid_search = dict(enable=False),
    
    output = dict(
        root='outputs/cocoop_dept',  
        result='results/cocoop_dept',
        remove_dirs=['root'],       
    ),
)

kgcoop_dept = dict(
    gpu_ids = [0],
    mode='b2n',
    
    train = dict(
        trainer='ExtrasLinearProbeKgCoOp',              
        cfg='vit_b16_ep10_ctxv1_bs4_lr35', 
        seeds=[1, 2, 3],             
        loadep=-1,                   
        shots=16,                   
        opts=[],      
    ),
    
    grid_search = dict(enable=False),
    
    output = dict(
        root='outputs/kgcoop_dept',  
        result='results/kgcoop_dept',
        remove_dirs=['root'],       
    ),
)

maple_dept = dict(
    gpu_ids = [0],
    mode='b2n',
    
    train = dict(
        trainer='ExtrasLinearProbeMaPLe',              
        cfg='vit_b16_c2_ep10_batch4_2ctx', 
        seeds=[1, 2, 3],             
        loadep=-1,                   
        shots=16,                   
        opts=[],      
    ),
    
    grid_search = dict(enable=False),
    
    output = dict(
        root='outputs/maple_dept',  
        result='results/maple_dept',
        remove_dirs=['root'],       
    ),
)

def get_config(name):
    cfg = copy.deepcopy(base)
    extend_cfg = copy.deepcopy(globals()[name])
    cfg.update(extend_cfg)
    return cfg
