class Config(object):
    #model config
    MODEL_NAME = 'UNet'
    UNET_IN = 3
    OUTPUT_STRIDE = 16
    ASPP_OUTDIM = 256
    SHORTCUT_DIM = 48
    SHORTCUT_KERNEL = 1
    NUM_CLASS = 8
    PRETRAIN = False
    TRAIN_BATCH_SIZE = 32
    VAL_BATCH_SIZE = 1

    #train config
    EPOCH = 50
    BASE_LR = 0.002
    WEIGHT_DECAY = 1e-5
    MODEL_SAVE_PATH = 'model param'
    LOG_PATH = 'logs'

    #predict config
    TEST_PATH = 'predict'

    #other config
    SIZE = (256, 256)