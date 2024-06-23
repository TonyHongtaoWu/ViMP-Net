# model settings
model = dict(
    type='DrainNet',
    generator=dict(
        type='ViMPNet',
        mid_channels=64,
        num_blocks=9,
        spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
        'basicvsr/spynet_20210409-c6c1bd09.pth'),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'),
)
# model training and testing settings
train_cfg = dict(fix_iter=5000)
test_cfg = dict(metrics=['PSNR'], crop_border=0)

train_dataset_type = 'SRFolderMultipleGTDataset'
val_dataset_type = 'SRFolderMultipleGTDataset'

train_pipeline = [
    dict(
        type='GenerateSegmentIndices',
        interval_list=[1],
        filename_tmpl='{08d}.png',
        start_idx=0),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='mask',
        flag='grayscale',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='drop',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='streak',
        flag='grayscale',
        channel_order='rgb'),

    dict(type='RescaleToZeroOne', keys=['lq', 'gt', 'mask', 'drop', 'streak']),
    dict(type='PairedRandomCrop', gt_patch_size=256),
    dict(
        type='Flip', keys=['lq', 'gt', 'mask', 'drop', 'streak'], flip_ratio=0.5,
        direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt', 'mask', 'drop', 'streak'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt', 'mask', 'drop', 'streak'], transpose_ratio=0.5),
    dict(type='FramesToTensor', keys=['lq', 'gt', 'mask', 'drop', 'streak']),
    dict(type='Collect', keys=['lq', 'gt', 'mask', 'drop', 'streak'], meta_keys=['lq_path', 'gt_path', 'mask_path', 'drop_path', 'streak_path'])
]

test_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1], filename_tmpl='{:08d}.png'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path', 'key'])
]

data = dict(
    workers_per_gpu=5,
    train_dataloader=dict(samples_per_gpu=1, drop_last=True), 
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),

    # train
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=train_dataset_type,
            lq_folder="../data/lq",
            gt_folder="../data/gt",
            mask_folder="../data/01drop",
            drop_folder="../data/drop",
            streak_folder="../data/mstreak",
            pipeline=train_pipeline,
            scale=1,
            num_input_frames=5,
            test_mode=False)),

    # # test
    test=dict(
        type=val_dataset_type,
        lq_folder='../data/testlq',
        gt_folder='../data/testgt',
        pipeline=test_pipeline,
        scale=1,
        #val_partition='official',
        test_mode=True),
)

# optimizer
optimizers = dict(
    generator=dict(
        type='Adam',
        lr=1e-4,
        betas=(0.9, 0.99),
        paramwise_cfg=dict(custom_keys={'spynet': dict(lr_mult=0.25)})))

# learning policy
total_iters = 500000
lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    periods=[500000],
    restart_weights=[1],
    min_lr=1e-9)

checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)

log_config = dict(
    interval=1000,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'

work_dir = f'../ViMPNettrain/'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
#1.5hole
