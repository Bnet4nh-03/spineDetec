#!/usr/bin/env python
# coding: utf-8

# In[7]:


from google.colab import drive
drive.mount("/content/drive", force_remount=True)


# In[8]:


CONFIG = {
    # ========== PATHS ==========
    # Data paths
    'train_images': '/content/drive/MyDrive/DATA/scoliosis2.v3i.coco/train/Images',
    'train_annotations': '/content/drive/MyDrive/DATA/scoliosis2.v3i.coco/train/_annotations.coco.json',
    'val_images': '/content/drive/MyDrive/DATA/scoliosis2.v3i.coco/valid/Images',
    'val_annotations': '/content/drive/MyDrive/DATA/scoliosis2.v3i.coco/valid/_annotations.coco.json',
    'test_images': '/content/drive/MyDrive/DATA/scoliosis2.v3i.coco/test/Images',
    'test_annotations': '/content/drive/MyDrive/DATA/scoliosis2.v3i.coco/test/_annotations.coco.json',

    # Output paths
    'output_dir': '/content/drive/MyDrive/RESULTS',
    'model_save_path': '/content/drive/MyDrive/RESULTS/best_spine_curve_model.pth',
    'checkpoint_dir': '/content/drive/MyDrive/RESULTS/checkpoints',
    'visualization_dir': '/content/drive/MyDrive/RESULTS/visualizations',
    'batch_prediction_dir': '/content/drive/MyDrive/RESULTS/batch_predictions',
    'logs_dir': '/content/drive/MyDrive/RESULTS/logs',

    # Test sample
    'test_image_sample': '/content/drive/MyDrive/DATA/AIspinecheck.v6i.coco/test/Images/258_png.rf.bea2b3ca0b6bc93d634888d4b709cc63.jpg',

    # ========== MODEL PARAMETERS ==========
    'backbone': 'efficientnet_b2',  # Options: efficientnet_b0/b1/b2/b3/b4, resnet50, etc.
    'pretrained': True,
    'input_channels': 3,
    'output_channels': 1,

    # ========== TRAINING PARAMETERS ==========
    'image_size': 480,  # Input image size (512, 384, 640, etc.)
    'batch_size': 8,  # Adjust based on GPU memory
    'num_workers': 4,  # DataLoader workers
    'epochs': 100,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'grad_clip': 1.0,

    # Scheduler parameters
    'scheduler_type': 'cosine_warmrestarts',  # Options: cosine_warmrestarts, step, plateau
    'scheduler_T0': 15,  # For CosineAnnealingWarmRestarts
    'scheduler_T_mult': 2,
    'scheduler_eta_min': 1e-6,

    # Early stopping
    'early_stopping': True,
    'patience': 20,

    # Mixed precision
    'use_amp': True,  # Automatic Mixed Precision

    # ========== LOSS PARAMETERS ==========
    'use_dice_loss': True,
    'use_focal_loss': True,
    'use_deep_supervision': True,
    'deep_supervision_weight': 0.3,
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,

    # ========== CURVE GENERATION PARAMETERS ==========
    'curve_thickness': 10,  # Thickness of curve in pixels for mask generation
    'num_curve_points': 200,  # Number of points on smooth B-spline curve
    'curve_smoothing': 0.5,  # Smoothing factor for spline (0=interpolate, higher=smoother)
    'spline_degree': 3,  # Degree of B-spline (1=linear, 2=quadratic, 3=cubic)

    # ========== AUGMENTATION PARAMETERS ==========
    # Light augmentation (warm-up & fine-tune phases)
    'aug_light_rotate': 8,
    'aug_light_scale': 0.05,
    'aug_light_shift': 0.05,
    'aug_light_brightness': 0.1,
    'aug_light_contrast': 0.1,

    # Medium augmentation
    'aug_medium_rotate': 15,
    'aug_medium_scale': 0.12,
    'aug_medium_shift': 0.08,
    'aug_medium_brightness': 0.2,
    'aug_medium_contrast': 0.2,

    # Strong augmentation (main training phase)
    'aug_strong_rotate': 25,
    'aug_strong_scale': 0.2,
    'aug_strong_shift': 0.12,
    'aug_strong_brightness': 0.3,
    'aug_strong_contrast': 0.3,
    'aug_strong_elastic_alpha': 50,
    'aug_strong_grid_distort': 0.3,

    # Augmentation schedule (as fraction of total epochs)
    'aug_schedule': {
        'light_warmup': 0.15,      # 0-15%: Light
        'medium_rampup': 0.35,     # 15-35%: Medium
        'strong_main': 0.70,       # 35-70%: Strong
        'medium_stabilize': 0.85,  # 70-85%: Medium
        'light_finetune': 1.0,     # 85-100%: Light
    },

    # ========== INFERENCE PARAMETERS ==========
    'prediction_threshold': 0.5,  # Threshold for binary mask
    'use_tta': True,  # Test-Time Augmentation
    'tta_scales': [1.0, 1.1],  # Multi-scale TTA
    'tta_flips': True,  # Horizontal flip TTA

    # Post-processing
    'morphology_kernel_size': 5,
    'min_curve_points': 3,  # Minimum points to fit curve
    'skeleton_iterations': 2,

    # ========== DEVICE PARAMETERS ==========
    'device': 'cuda',  # 'cuda' or 'cpu'
    'gpu_id': 0,  # GPU device ID
    'pin_memory': True,
    'benchmark': True,  # cudnn.benchmark

    # ========== LOGGING & CHECKPOINTING ==========
    'save_frequency': 5,  # Save checkpoint every N epochs
    'log_frequency': 10,  # Log metrics every N batches
    'visualize_frequency': 1,  # Visualize predictions every N epochs
    'num_visualize_samples': 4,  # Number of samples to visualize

    # ========== EVALUATION PARAMETERS ==========
    'compute_metrics': True,
    'metrics': ['dice', 'iou', 'precision', 'recall', 'f1'],
    'save_predictions': True,
    'batch_test_size': 5,  # Number of images for batch testing

    # ========== REPRODUCIBILITY ==========
    'seed': 42,
    'deterministic': False,  # Set to True for reproducibility (slower)

    # ========== EXPERIMENT TRACKING ==========
    'experiment_name': 'spine_curve_detection_v1',
    'run_name': None,  # Auto-generated if None
    'tags': ['spine', 'curve', 'segmentation', 'coco'],
    'notes': 'Spine curve detection with dynamic augmentation',

    'lr' : 1e-3,
    'num_point_to_smoth' : 9,
    'smooth_factor': 50,
}


# In[9]:


get_ipython().system('jupyter nbconvert --to python "/content/drive/MyDrive/Colab Notebooks/CONFIG.ipynb"')

