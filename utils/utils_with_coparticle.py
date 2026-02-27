# imports
import inspect
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.style as style
import cv2
import datetime
import os
import json
import imageio
import fnmatch
import zipfile
from datetime import datetime
import csv 
import pandas as pd 


# torch
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.ops as ops
from typing import Tuple
from torch.utils.data import DataLoader


matplotlib.use("Agg")

def get_config(fpath):
    with open(fpath, 'r') as f:
        config = json.load(f)
    return config


def load_model(config_path='./configs/balls.json'):
    # load config
    try:
        config = get_config(config_path)
    except FileNotFoundError:
        raise SystemExit("config file not found")
    hparams = config  # to save a copy of the hyper-parameters
    device = config['device']
    if 'cuda' in device:
        device = torch.device(f'{device}' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
        


    # data and general
    ds = config['ds']
    ch = config['ch']  # image channels
    image_size = config['image_size']
    n_views = config.get('n_views', 1)
    views = config.get('views',['front'])
    root = config['root']  # dataset root
    run_prefix = config['run_prefix']
    load_model = config['load_model']
    pretrained_path = config['pretrained_path']  # path of pretrained model to load, if None, train from scratch

    # model
    timestep_horizon = config['timestep_horizon']
    pad_mode = config['pad_mode']
    n_kp_per_patch = config['n_kp_per_patch']  # kp per patch in prior, best to leave at 1
    n_kp_prior = config['n_kp_prior']  # number of prior kp to filter for the kl
    n_kp_enc = config['n_kp_enc']  # total posterior kp
    patch_size = config['patch_size']  # prior patch size
    anchor_s = config['anchor_s']  # posterior patch/glimpse ratio of image size

    # visual latent features
    features_dist = config.get('features_dist', 'gauss')
    learned_feature_dim = config['learned_feature_dim']
    learned_bg_feature_dim = config.get('learned_bg_feature_dim', learned_feature_dim)
    n_fg_categories = config.get('n_fg_categories', 8)  # Number of foreground feature categories (if categorical)
    n_fg_classes = config.get('n_fg_classes', 4)  # Number of foreground feature classes per category
    n_bg_categories = config.get('n_bg_categories', 4)  # Number of background feature categories
    n_bg_classes = config.get('n_bg_classes', 4)

    # latent context
    context_dist = config.get('context_dist', 'gauss')
    context_dim = config['context_dim']
    ctx_pool_mode = config.get("ctx_pool_mode", "none")
    n_ctx_categories = config.get('n_ctx_categories', 8)  # Number of context feature categories (if categorical)
    n_ctx_classes = config.get('n_ctx_classes', 4)  # Number of context feature classes per category

    # latent_actions 
    learned_action_feature_dim = config.get('learned_action_feature_dim', learned_feature_dim)

    dropout = config['dropout']
    use_resblock = config['use_resblock']

    # optimization
    batch_size = config['batch_size']
    lr = config['lr']
    num_epochs = config['num_epochs']
    start_epoch = config.get('start_epoch', 0)
    weight_decay = config['weight_decay']
    adam_betas = config['adam_betas']
    adam_eps = config['adam_eps']
    use_scheduler = config['use_scheduler']
    scheduler_gamma = config['scheduler_gamma']
    warmup_epoch = config['warmup_epoch']
    recon_loss_type = config['recon_loss_type']
    beta_kl = config['beta_kl']
    beta_dyn = config['beta_dyn']
    beta_rec = config['beta_rec']
    beta_dyn_rec = config['beta_dyn_rec']
    beta_obj = config.get('beta_obj', 0.0)
    kl_balance = config['kl_balance']  # balance between visual features and the other particle attributes
    num_static_frames = config['num_static_frames']  # frames for which kl is calculated w.r.t constant prior params

    # priors
    scale_std = config['scale_std']
    offset_std = config['offset_std']
    obj_on_alpha = config['obj_on_alpha']  # transparency beta distribution "a"
    obj_on_beta = config['obj_on_beta']  # transparency beta distribution "b"

    # evaluation
    eval_epoch_freq = config['eval_epoch_freq']
    eval_im_metrics = config['eval_im_metrics']
    cond_steps = config['cond_steps']  # conditional frames for the dynamics module during inference
    ctx_for_eval = config.get('ctx_for_eval', False)

    # visualization
    iou_thresh = config['iou_thresh']  # threshold for NMS for plotting bounding boxes
    topk = min(config['topk'], config['n_kp_enc'])  # top-k particles to plot
    animation_horizon = config['animation_horizon']

    # transformer - PINT
    pint_enc_layers = config['pint_enc_layers']
    pint_enc_heads = config['pint_enc_heads']
    pint_ctx_layers = config['pint_ctx_layers']
    pint_ctx_heads = config['pint_ctx_heads']
    pint_dyn_layers = config['pint_dyn_layers']
    pint_dyn_heads = config['pint_dyn_heads']
    pint_dim = config['pint_dim']

    predict_delta = config['predict_delta']  # dynamics module predicts the delta from previous step

    normalize_rgb = config['normalize_rgb']
    obj_res_from_fc = config["obj_res_from_fc"]
    obj_ch_mult = config["obj_ch_mult"]
    obj_ch_mult_prior = config.get("obj_ch_mult_prior", obj_ch_mult)
    obj_base_ch = config["obj_base_ch"]
    obj_final_cnn_ch = config["obj_final_cnn_ch"]
    bg_res_from_fc = config["bg_res_from_fc"]
    bg_ch_mult = config["bg_ch_mult"]
    bg_base_ch = config["bg_base_ch"]
    bg_final_cnn_ch = config["bg_final_cnn_ch"]
    num_res_blocks = config["num_res_blocks"]
    cnn_mid_blocks = config.get('cnn_mid_blocks', False)
    mlp_hidden_dim = config.get('mlp_hidden_dim', 256)
    use_ep_done_mask = config.get('ep_done_mask', False)

    # actions
    action_condition = config.get('action_condition', False)
    action_dim = config.get('action_dim', 0)
    null_action_embed = config.get('null_action_embed', False)

    random_action_condition = config.get('random_action_condition', False)
    random_action_dim = config.get('random_action_dim', 0)

    # language
    language_condition = config.get('language_condition', False)
    language_embed_dim = config.get('language_embed_dim', 0)
    language_max_len = config.get('language_max_len', 32)

    # image goal condition
    img_goal_condition = config.get('image_goal_condition', False)

    tiny = config.get('tiny',False)
    dense = config.get('dense',False)
    # load data
    # dataset = get_video_dataset(ds, root, seq_len=timestep_horizon+1, mode='train', image_size=image_size,tiny=tiny,dense=dense)
    # dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, 
    #                         num_workers=0,  # NOTE SUPRESSING PARALLELISM FOR DEBUGGING / also, overhead doesn't appear to be worth it
    #                         pin_memory=True,
    #                         drop_last=True)
    # action 
    action_generation = config.get('action_generation', False)
    learned_policy_action_feature_dim = config.get('learned_policy_action_feature_dim', 7)
    action_latent_dim = config.get('learned_action_feature_dim',8)

    # model
    return DLP(cdim=ch,  # Number of input image channels
                image_size=image_size,  # Input image size (assumed square)
                normalize_rgb=normalize_rgb,  # If True, normalize RGB to [-1, 1], else keep [0, 1]
                n_views=n_views,  # number of input views (e.g., multiple cameras)

                # Keypoint and patch configuration
                n_kp_per_patch=n_kp_per_patch,  # Number of proposal/prior keypoints to extract per patch
                patch_size=patch_size,  # Size of patches for keypoint proposal network
                anchor_s=anchor_s,  # Glimpse size ratio relative to image size
                n_kp_enc=n_kp_enc,  # Number of posterior keypoints to learn
                n_kp_prior=n_kp_prior,  # Number of keypoints to filter from prior proposals

                # Network configuration
                pad_mode=pad_mode,  # Padding mode for CNNs ('zeros' or 'replicate')
                dropout=dropout,  # Dropout rate for transformers

                # Feature representation
                features_dist=features_dist,  # Distribution type for features ('gauss' or 'categorical')
                learned_feature_dim=learned_feature_dim,  # Dimension of learned visual features
                learned_bg_feature_dim=learned_bg_feature_dim,
                # Background feature dimension (if None, equals learned_feature_dim)
                n_fg_categories=n_fg_categories,  # Number of foreground feature categories (if categorical)
                n_fg_classes=n_fg_classes,  # Number of foreground feature classes per category
                n_bg_categories=n_bg_categories,  # Number of background feature categories
                n_bg_classes=n_bg_classes,  # Number of background feature classes per category

                # Prior distributions parameters
                scale_std=scale_std,  # Prior standard deviation for scale
                offset_std=offset_std,  # Prior standard deviation for offset
                obj_on_alpha=obj_on_alpha,  # Alpha parameter for transparency Beta distribution
                obj_on_beta=obj_on_beta,  # Beta parameter for transparency Beta distribution

                # Object decoder architecture
                obj_res_from_fc=obj_res_from_fc,  # Initial resolution for object encoder-decoder
                obj_ch_mult_prior=obj_ch_mult_prior,  # Channel multipliers for prior patch encoder (kp proposals)
                obj_ch_mult=obj_ch_mult,  # Channel multipliers for object encoder-decoder
                obj_base_ch=obj_base_ch,  # Base channels for object encoder-decoder
                obj_final_cnn_ch=obj_final_cnn_ch,  # Final CNN channels for object encoder-decoder

                # Background decoder architecture
                bg_res_from_fc=bg_res_from_fc,  # Initial resolution for background encoder-decoder
                bg_ch_mult=bg_ch_mult,  # Channel multipliers for background encoder-decoder
                bg_base_ch=bg_base_ch,  # Base channels for background encoder-decoder
                bg_final_cnn_ch=bg_final_cnn_ch,  # Final CNN channels for background encoder-decoder

                # Network architecture options
                use_resblock=use_resblock,  # Use residual blocks in encoders-decoders
                num_res_blocks=num_res_blocks,  # Number of residual blocks per resolution
                cnn_mid_blocks=cnn_mid_blocks,  # Use middle blocks in CNN
                mlp_hidden_dim=mlp_hidden_dim,  # Hidden dimension for MLPs

                # Particle interaction transformer (PINT) configuration
                pint_enc_layers=pint_enc_layers,  # Number of PINT encoder layers
                pint_enc_heads=pint_enc_heads,  # Number of PINT encoder attention heads

                # Dynamics configuration
                timestep_horizon=timestep_horizon,  # Number of timesteps to predict ahead
                n_static_frames=num_static_frames,  # Number of initial frames for static KL optimization
                predict_delta=predict_delta,  # Predict position deltas instead of absolute positions
                context_dim=context_dim,  # Context latent dimension (if None, equals learned_feature_dim)
                ctx_dist=context_dist,  # Context distribution type ('gauss' or 'categorical')
                n_ctx_categories=n_ctx_categories,  # Number of context categories (if categorical)
                n_ctx_classes=n_ctx_classes,  # Number of context classes per category
                ctx_pool_mode=ctx_pool_mode,  # Context pooling mode ('none' = per-particle context)

                # Context and dynamics transformer configuration
                pint_dyn_layers=pint_dyn_layers,  # Number of dynamics transformer layers
                pint_dyn_heads=pint_dyn_heads,  # Number of dynamics transformer heads
                pint_dim=pint_dim,  # Hidden dimension for PINT
                pint_ctx_layers=pint_ctx_layers,  # Number of context transformer layers
                pint_ctx_heads=pint_ctx_heads,

                # external conditioning
                action_condition=action_condition,  # condition on actions
                action_dim=action_dim,  # dimension of input actions
                null_action_embed=null_action_embed,
                random_action_condition=random_action_condition,
                random_action_dim=random_action_dim,
                # learn a "no-input-action" embedding, to learn on action-free videos as well
                language_condition=language_condition,  # condition on language embedding
                language_embed_dim=language_embed_dim,  # embedding dimension for each token
                language_max_len=language_max_len,  # maximum tokens per prompt
                img_goal_condition=img_goal_condition,  # condition the future on image goal
                 # --- NEW: action-reward-generation --- #
                action_generation=action_generation,  # whether to generate actions (action-as-particle)
                action_latent_dim=action_latent_dim,  # the latent dimension of encoded actions
                # other action parameters will be taken from the `external conditioning` part above
                # setting this to True will also set `action_condition=True`, will use `action_dim` as well
                reward_generation=False,  # whether to generate rewards (reward-as-particle)
                # reward_dim=reward_dim,  # reward original dimension
                # reward_latent_dim=reward_latent_dim,  # the latent dimension of encoded rewards
                ).to(device)

def plot_actions(gen_acts,gt_acts,timestep_horizon,ndim=8,id=None,root="experiments"):
    # plot difference in l2 norm over time (frames)
    # action_rec : [bs, timesteps, dim]
    X = torch.arange(timestep_horizon)
    # Y = action_rec[sample].squeeze() # average action prediction across all sample  
    # nframes, ndim = Y.shape
    
    fig,axs = plt.subplots(ndim,1)
    

    

    
    gen_acts = gen_acts.cpu()
    gt_acts = gt_acts.cpu()
    # SAVE QUANTATIVE METRICS 
    diff = gen_acts-gt_acts
    # average the difference across timesteps for each dimension seprately
    l2 = torch.linalg.vector_norm(diff,dim=0)
    min = torch.min(diff,dim=0)[0]
    max = torch.max(diff,dim=0)[0]
    median = torch.median(diff,dim=0)[0]
    mean = torch.mean(diff,dim=0)
    var = torch.var(diff,dim=0)
    # assert torch.all(var >= 0), torch.all(l2>=0)
    stats_array = torch.concatenate([l2,min,max,median,mean,var])
    stats_array = stats_array.unsqueeze(0).cpu().numpy()

    stats_path = os.path.join(root,'action_recon_stats.csv')
    if os.path.exists(stats_path):
        stats_df = pd.DataFrame(stats_array)
        stats_df.to_csv(stats_path,mode='a', header=False,index=False)
    else:
        # introduce header if this is the first write
        stat_names = ["l2","min","max","median","mean", "var"]
        stats = []
        for stat in stat_names:
            stats += [stat + f"_dim_{dim}" for dim in range(ndim)]
        stats_df = pd.DataFrame(stats_array,columns=stats)
        stats_df.to_csv(stats_path,mode='a', index=False)
   
    
    
    
    # SAVE VISUALIZAITONS  
    for idx in range(ndim):
        axs[idx].plot(X,gen_acts[:,idx].squeeze().cpu().detach(),color='red',lw=0.4,alpha=0.5,label="generated")
        axs[idx].plot(X,gt_acts[:,idx].squeeze().cpu().detach(),color='blue',lw=0.4,alpha=0.5,label="ground truth")
    
    
    
    # plt.plot(X,Y.mean(axis=0).cpu().detach(),color='red',lw=1,label ='mean')
    plt.legend()
    # plt.title('L2 Norm of A-hat vs. A')
    id = id if id is not None else datetime.now() 
    dir = os.path.join(root,'action_recon')
    os.makedirs(dir,exist_ok=True)
    plt.savefig(os.path.join(dir,f'{id}.png'))
    plt.close()
    
    # save stats about the cycles 
    # stats_path = os.path.join(root,'action_recon_stats.csv')
    # if os.path.exists(stats_path):
    #     stats = stats[1:]
    # with open(stats_path, mode='a', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(stats)
           
def animate_trajectories(orig_trajectory, pred_trajectory, pred_trajectory_2=None, path='./traj_anim.gif',
                         duration=4 / 50, rec_to_pred_t=10, title=None, t1='', t2='', goal_img=None,
                         orig_trajectory2=None, pred_trajectory_12=None, pred_trajectory_22=None, goal_img2=None):
    
    # rec_to_pred_t: the timestep from which prediction transitions from reconstruction to generation
    # goal_img: np array: [h, w, ch]
    # prepare images
    font = cv2.FONT_HERSHEY_SIMPLEX
    origin = (5, 15)
    fontScale = 0.4
    color = (255, 255, 255)
    gt_border_color = (255, 0, 0)
    rec_border_color = (0, 0, 255)
    gen_border_color = (0, 255, 0)
    border_size = 2
    thickness = 1
    gt_traj_prep = []
    pred_traj_prep = []
    pred_traj_prep_2 = []

    # second view
    gt_traj_prep2 = []
    pred_traj_prep12 = []
    pred_traj_prep_22 = []

    goal_img_prep = None
    if goal_img is not None:
        goal_img_prep = (goal_img.clip(0, 1) * 255).astype(np.uint8).copy()
        # Add border to goal image
        goal_img_prep = cv2.copyMakeBorder(goal_img_prep, border_size, border_size, border_size, border_size,
                                           cv2.BORDER_CONSTANT, value=(128, 128, 128))  # Gray border for goal
        if goal_img2 is not None:
            goal_img_prep2 = (goal_img2.clip(0, 1) * 255).astype(np.uint8).copy()
            # Add border to goal image
            goal_img_prep2 = cv2.copyMakeBorder(goal_img_prep2, border_size, border_size, border_size, border_size,
                                                cv2.BORDER_CONSTANT, value=(128, 128, 128))  # Gray border for goal

        # Create "Goal" text plate
        goal_text_color = (0, 0, 0)
        goal_fontScale = 0.4
        goal_thickness = 1
        goal_font = cv2.FONT_HERSHEY_SIMPLEX
        goal_text_h = 20
        goal_text_w = 50
        goal_text_plate = (np.ones((goal_text_h, goal_text_w, 3)) * 255).astype(np.uint8)
        goal_text_origin = (5, goal_text_h // 2 + 5)
        goal_text_plate = cv2.putText(goal_text_plate, 'Goal', goal_text_origin, goal_font,
                                      goal_fontScale, goal_text_color, goal_thickness, cv2.LINE_AA)

    for i in range(orig_trajectory.shape[0]):
        image = (orig_trajectory[i] * 255).astype(np.uint8).copy()
        image = cv2.putText(image, f'GT:{i}', origin, font, fontScale, color, thickness, cv2.LINE_AA)
        # add border
        image = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT,
                                   value=gt_border_color)
        gt_traj_prep.append(image)

        text = f'REC:{i}' if i < rec_to_pred_t else f'P{t1}:{i}'
        image = (pred_trajectory[i].clip(0, 1) * 255).astype(np.uint8).copy()
        image = cv2.putText(image, text, origin, font, fontScale, color, thickness, cv2.LINE_AA)
        # add border
        border_color = rec_border_color if i < rec_to_pred_t else gen_border_color
        image = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT,
                                   value=border_color)
        pred_traj_prep.append(image)

        if pred_trajectory_2 is not None:
            text = f'REC:{i}' if i < rec_to_pred_t else f'P{t2}:{i}'
            image = (pred_trajectory_2[i].clip(0, 1) * 255).astype(np.uint8).copy()
            image = cv2.putText(image, text, origin, font, fontScale, color, thickness, cv2.LINE_AA)
            # add border
            border_color = rec_border_color if i < rec_to_pred_t else gen_border_color
            image = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT,
                                       value=border_color)
            pred_traj_prep_2.append(image)

        if orig_trajectory2 is not None:
            image = (orig_trajectory2[i] * 255).astype(np.uint8).copy()
            image = cv2.putText(image, f'GT:{i}', origin, font, fontScale, color, thickness, cv2.LINE_AA)
            # add border
            image = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT,
                                       value=gt_border_color)
            gt_traj_prep2.append(image)

        if pred_trajectory_12 is not None:
            text = f'REC:{i}' if i < rec_to_pred_t else f'P{t1}:{i}'
            image = (pred_trajectory_12[i].clip(0, 1) * 255).astype(np.uint8).copy()
            image = cv2.putText(image, text, origin, font, fontScale, color, thickness, cv2.LINE_AA)
            # add border
            border_color = rec_border_color if i < rec_to_pred_t else gen_border_color
            image = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT,
                                       value=border_color)
            pred_traj_prep12.append(image)

        if pred_trajectory_22 is not None:
            text = f'REC:{i}' if i < rec_to_pred_t else f'P{t2}:{i}'
            image = (pred_trajectory_22[i].clip(0, 1) * 255).astype(np.uint8).copy()
            image = cv2.putText(image, text, origin, font, fontScale, color, thickness, cv2.LINE_AA)
            # add border
            border_color = rec_border_color if i < rec_to_pred_t else gen_border_color
            image = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT,
                                       value=border_color)
            pred_traj_prep_22.append(image)

    total_images = []
    for i in range(len(orig_trajectory)):
        white_border = (np.ones((gt_traj_prep[i].shape[0], 4, gt_traj_prep[i].shape[-1])) * 255).astype(np.uint8)
        if pred_trajectory_2 is not None:
            concat_img = np.concatenate([gt_traj_prep[i],
                                         white_border,
                                         pred_traj_prep[i],
                                         white_border,
                                         pred_traj_prep_2[i]], axis=1)
        else:
            concat_img = np.concatenate([gt_traj_prep[i],
                                         white_border,
                                         pred_traj_prep[i]], axis=1)
        if orig_trajectory2 is not None:
            white_separator = (np.ones((8, concat_img.shape[1], 3)) * 255).astype(np.uint8)
            white_border = (np.ones((gt_traj_prep2[i].shape[0], 4, gt_traj_prep2[i].shape[-1])) * 255).astype(np.uint8)
            if pred_trajectory_22 is not None:
                concat_img2 = np.concatenate([gt_traj_prep2[i],
                                              white_border,
                                              pred_traj_prep12[i],
                                              white_border,
                                              pred_traj_prep_22[i]], axis=1)
            else:
                concat_img2 = np.concatenate([gt_traj_prep2[i],
                                              white_border,
                                              pred_traj_prep12[i]], axis=1)
            concat_img = np.concatenate([concat_img, white_separator, concat_img2], axis=0)

        # Add title if provided
        if title is not None:
            text_color = (0, 0, 0)
            fontScale = 0.25
            thickness = 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            h = 25
            w = concat_img.shape[1]
            text_plate = (np.ones((h, w, 3)) * 255).astype(np.uint8)
            w_orig = orig_trajectory.shape[1] // 2
            origin = (w_orig // 6, h // 2)
            text_plate = cv2.putText(text_plate, title, origin, font, fontScale, text_color, thickness,
                                     cv2.LINE_AA)
            concat_img = np.concatenate([text_plate, concat_img], axis=0)

        # Add goal image if provided
        if goal_img is not None:
            mult_factor = 1 if goal_img2 is None else 2
            # Create white separator
            white_separator = (np.ones((8, concat_img.shape[1], 3)) * 255).astype(np.uint8)

            # Create goal section with text and image
            goal_section_width = concat_img.shape[1]
            goal_img_center_x = goal_section_width // 2 - (goal_img_prep.shape[1] * mult_factor) // 2

            # Create goal section background
            goal_section_height = goal_img_prep.shape[0]
            goal_section = (np.ones((goal_section_height, goal_section_width, 3)) * 255).astype(np.uint8)

            # Place goal text on the left
            text_x_pos = max(0, goal_img_center_x - goal_text_plate.shape[1] - 10)
            text_y_start = (goal_section_height - goal_text_plate.shape[0]) // 2
            text_y_end = text_y_start + goal_text_plate.shape[0]
            goal_section[text_y_start:text_y_end, text_x_pos:text_x_pos + goal_text_plate.shape[1]] = goal_text_plate

            # Place goal image in the center
            goal_section[:, goal_img_center_x:goal_img_center_x + goal_img_prep.shape[1]] = goal_img_prep

            if goal_img2 is not None:
                goal_section[:, goal_img_center_x + goal_img_prep.shape[1]:goal_img_center_x + 2 * goal_img_prep.shape[
                    1]] = goal_img_prep2

            # Concatenate everything
            concat_img = np.concatenate([concat_img, white_separator, goal_section], axis=0)

        total_images.append(concat_img)

    imageio.mimsave(path, total_images, duration=(1000 / duration), loop=0)  # 1/50

