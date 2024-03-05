from collections import defaultdict


class DatasetConfig:
    model_folder = 'models'
    train_folder = 'train'
    test_folder = 'test'
    img_folder = 'rgb'
    depth_folder = 'depth'
    img_ext = 'png'
    depth_ext = 'png'
    mask_ext = 'png'


config = defaultdict(lambda *_: DatasetConfig())

config['tless'] = tless = DatasetConfig()
tless.model_folder = 'models_cad'
tless.test_folder = 'test_primesense'
tless.train_folder = 'train_primesense'

config['hb'] = hb = DatasetConfig()
hb.test_folder = 'test_primesense'

config['itodd'] = itodd = DatasetConfig()
itodd.depth_ext = 'tif'
itodd.img_folder = 'gray'
itodd.img_ext = 'tif'

config['mvpsp'] = mvpsp = DatasetConfig()
mvpsp.model_folder = 'models_eval'  # use downsampled models for training to speed up surface sampling
config['mvpsp_orx'] = mvpsp_orx = DatasetConfig()
mvpsp_orx.model_folder = mvpsp.model_folder
mvpsp_orx.test_folder = 'test_orx'
mvpsp_orx.img_ext = 'jpg'
config['mvpsp_pbr_random_lighting'] = mvpsp_pbr_random_lighting = DatasetConfig()
mvpsp_pbr_random_lighting.model_folder = mvpsp.model_folder
mvpsp_pbr_random_lighting.train_folder = 'train_pbr_random_lighting'
mvpsp_pbr_random_lighting.test_folder = 'test_pbr_random_lighting'
mvpsp_pbr_random_lighting.img_ext = 'jpg'
config['mvpsp_pbr_fixed_lighting'] = mvpsp_pbr_fixed_lighting = DatasetConfig()
mvpsp_pbr_fixed_lighting.model_folder = mvpsp.model_folder
mvpsp_pbr_fixed_lighting.train_folder = 'train_pbr_fixed_lighting'
mvpsp_pbr_fixed_lighting.test_folder = 'test_pbr_fixed_lighting'
mvpsp_pbr_fixed_lighting.img_ext = 'jpg'
config['mvpsp_pbr_random_texture'] = mvpsp_pbr_random_texture = DatasetConfig()
mvpsp_pbr_random_texture.model_folder = mvpsp.model_folder
mvpsp_pbr_random_texture.train_folder = 'train_pbr_random_texture'
mvpsp_pbr_random_texture.test_folder = 'test_pbr_random_texture'
mvpsp_pbr_random_texture.img_ext = 'jpg'
