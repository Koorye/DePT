import copy


base = dict(
    # dataset configs
    data = dict(
        root='/ai/zhangji/wuhao/koorye/Datasets',
        datasets_base_to_new=['dtd', 'caltech101', 'eurosat', 'ucf101', 'oxford_flowers', 
                              'oxford_pets', 'stanford_cars', 'fgvc_aircraft', 'food101', 'sun397', 'imagenet'],
        # datasets_base_to_new=['dtd'],
        datasets_cross_dataset=['caltech101', 'oxford_pets', 'stanford_cars', 'oxford_flowers', 'food101',
                                'fgvc_aircraft', 'sun397', 'dtd', 'eurosat', 'ucf101',
                                'imagenetv2', 'imagenet_sketch', 'imagenet_a', 'imagenet_r'],
    ),

    # mail configs
    mail = dict(
        username='1311586225@qq.com',
        password='ftzcotjfnflwbabb',
        host='smtp.qq.com',
        to='1311586225@qq.com',
    ),
)

##########################################################

elpcoop_only_img = dict(
    # GPU configs
    gpu_ids = [0, 1, 2, 3, 4, 5],
    mode='b2n',
    
    # training configs
    train = dict(
        trainer='ExtrasLinearProbeCoOp',
        cfg='vit_b16_ep10_bs4_lr35',
        seeds=[1, 2, 3],
        loadep=-1,
        shots=16,
        opts=[],
    ),
    
    # grid search configs
    grid_search = dict(enable=False),
    
    # output configs
    output = dict(
        root='outputs/elpcoop_only_img',
        result='results/elpcoop_only_img',
        remove_dirs=['root'],
    ),
)


def get_config(name):
    cfg = copy.deepcopy(base)
    extend_cfg = copy.deepcopy(globals()[name])
    cfg.update(extend_cfg)
    return cfg
