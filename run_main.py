import argparse
import torch
from accelerate import Accelerator 
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from models import ATM

from data_provider.data_factory import data_provider
import time
import random
import numpy as np
import os

#if __name__ == '__main__':
os.environ['curl_ca_bundle'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'

from utils.tools import EarlyStopping, adjust_learning_rate, vali

fix_seed = 2025
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

if torch.cuda.is_available():
    print("cuda:", torch.cuda.is_available())
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("GPU count =", torch.cuda.device_count())
    torch.cuda.manual_seed(fix_seed)         # Set CUDA random seed
    torch.cuda.manual_seed_all(fix_seed)     # Set all GPU random seeds
    torch.backends.cudnn.deterministic = True  # Ensure that the calculation results are consistent with each run
    torch.backends.cudnn.benchmark = False     # Disable dynamic optimization

def get_args():
    parser = argparse.ArgumentParser(description='ATM')
    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model_comment', type=str, default='none', help='prefix when saving test results')
    parser.add_argument('--model', type=str, default='ATM',
                        help='model name, options: [ATM]')
    parser.add_argument('--model_path', type=str, default='', help='model path')
    parser.add_argument('--seed', type=int, default=2025, help='random seed')

    # data loader
    parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='../dataset/ETT-small/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; '
                             'M:multivariate predict multivariate, S: univariate predict univariate, '
                             'MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--loader', type=str, default='modal', help='dataset type')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, '
                             'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                             'you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=336, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=32, help='dimension of model')
    parser.add_argument('--inter_dim', type=int, default=32, help='intermediate dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--conv_layers', type=int, default=4, help='num of convolution layers')
    parser.add_argument('--num_experts', type=int, default=8, help='num of experts')
    parser.add_argument('--top_k', type=int, default=2, help='num of activated experts')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument('--n_querys', type=int, default=8, help='num of semantic querys')
    parser.add_argument('--threshold_ratio', type=float, default=0.995, help='threshold of patch similarity')
    parser.add_argument('--high_freq', type=float, default=1.0, help='coefficient of high frequence')
    parser.add_argument('--prob_bias', type=float, default=0.01, help='bias of probs')
    parser.add_argument('--prob_bias_end', type=int, default=150, help='the ending iter of bias of probs')
    parser.add_argument('--use_time_tokenizer', type=int, default=1, help='whether to use time_tokenizer, options:[0-no, 1-yes, 2-use MLP]')
    parser.add_argument('--aux_loss', type=int, default=1, help='whether to use auxiliary loss')
    parser.add_argument('--aux_loss_factor', type=float, default=0.05, help='auxiliary loss factor')
    parser.add_argument('--use_semantic_pe', type=int, default=1, help='whether to use semantic_pe')
    parser.add_argument('--use_moe', type=int, default=1, help='whether use MoE')
    parser.add_argument('--router_aux_loss_factor', type=float, default=0.02, help='router auxiliary loss factor')
    parser.add_argument('--apply_router_aux_loss', type=int, default=1, help='apply router auxiliary loss')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--head_dropout', type=float, default=0.1, help='head dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--model_verbose', type=int, default=1, help='whether output details')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
    parser.add_argument('--patience', type=int, default=4, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0025, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MAE', help='loss function, options:[MAE, Smooth_L1]')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--percent', type=int, default=100)
    parser.add_argument('--save_optim', type=int, default=0)

    args = parser.parse_args()
    return args

args = get_args()

# Set the gradients of the unlearned parameters to zero
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
#deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
#print("ddp_kwargs:",ddp_kwargs,", deepspeed_plugin:",deepspeed_plugin)

if torch.cuda.is_available():
    #accelerator = Accelerator(mixed_precision='no')
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs]) #deepspeed_plugin=deepspeed_plugin
else:
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], mixed_precision='no')
accelerator.print(f'device {str(accelerator.device)} is used!')

args.mixed_precision = torch.float32
if accelerator.mixed_precision == 'bf16':
   args.mixed_precision = torch.bfloat16
elif accelerator.mixed_precision == 'fp16':
    args.mixed_precision = torch.float16
print("mixed_precision:", args.mixed_precision, accelerator.mixed_precision)

if __name__ == '__main__':
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_ft{}_sl{}_ll{}_pl{}_dm{}_id{}_nh{}_pb{}_ed{}_nq{}_cl{}_ne{}_tk{}_dl{}_th{}_hf{}_af{}_dp{}_lr{}'.format(
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.inter_dim,
            args.n_heads,
            args.prob_bias,
            args.prob_bias_end,
            args.n_querys,
            args.conv_layers,
            args.num_experts,
            args.top_k,
            args.d_layers,
            args.threshold_ratio,
            args.high_freq,
            args.aux_loss_factor, 
            args.dropout,
            args.learning_rate)
        print(setting)
        print("train_epochs:", args.train_epochs)

        train_data, train_loader = data_provider(args, 'train')
        vali_data, vali_loader = data_provider(args, 'val')
        test_data, test_loader = data_provider(args, 'test')


        model_dicts = {'ATM': ATM}
        model = model_dicts[args.model].Model(args).float()

        # loading model
        if args.model_path != '':
            print("loading model...")
            model.load_state_dict(torch.load(args.model_path + '/checkpoint.pt', map_location=accelerator.device))

        #path = os.path.join(args.checkpoints,
        #                    setting + '-' + args.model_comment)  # unique checkpoint saving path
        # args.content = load_content(args)
        path = os.path.join(args.checkpoints, args.task_name + '/', args.data + '/')
        if not os.path.exists(path) and accelerator.is_local_main_process:
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        print("save_optim:", args.save_optim)
        early_stopping = EarlyStopping(accelerator, args.patience, verbose=True, save_optim=args.save_optim)

        trained_parameters = []
        for p in model.parameters():
            if p.requires_grad is True:
                trained_parameters.append(p)

        model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)
        #print("optim_state:", model_optim.state_dict())

        if args.lradj == 'COS':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
        else:
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=args.pct_start,
                                                epochs=args.train_epochs,
                                                max_lr=args.learning_rate)
        
        print("Loss:", args.loss)
        if args.loss == 'MSE':
            criterion = nn.MSELoss()
        elif args.loss == 'Smooth_L1':
            criterion = nn.SmoothL1Loss()
        else:
            criterion = nn.L1Loss()

        mse_metric = nn.MSELoss()
        mae_metric = nn.L1Loss()

        train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
            train_loader, vali_loader, test_loader, model, model_optim, scheduler)

        # if os.path.exists(args.model_path + '/optimizer.pt'):
        #     print("loading optimizer...")
        #     model_optim.load_state_dict(torch.load(args.model_path + '/optimizer.pt', map_location=accelerator.device))
        #if os.path.exists(args.model_path + '/scheduler.pt'):
        #    print("loading scheduler...")
        #    scheduler.load_state_dict(torch.load(args.model_path + '/scheduler.pt', map_location=accelerator.device))
        #print(scheduler.state_dict())
        print("model_optim:", model_optim)
        
        print("data_loader")

        if args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        print("start training...")
        for epoch in range(args.train_epochs):
            iter_count = 0
            train_loss = []
            train_mae_loss = []

            model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
                iter_count += 1
                # if iter_count == 2: break
                model_optim.zero_grad()

                batch_x = batch_x.float().to(accelerator.device) 
                batch_y = batch_y.float().to(accelerator.device)
                batch_x_mark = batch_x_mark.float().to(accelerator.device)
                batch_y_mark = batch_y_mark.float().to(accelerator.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(
                    accelerator.device)
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                    accelerator.device)

                #with accelerator.autocast():
                # encoder - decoder
                if args.use_amp:
                    with torch.cuda.amp.autocast():
                        if args.output_attention:
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if args.features == 'MS' else 0
                        outputs = outputs[:, -args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        #print(batch_x.dtype)
                        outputs, aux_loss = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if args.features == 'MS' else 0
                    outputs = outputs[:, -args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -args.pred_len:, f_dim:]
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
                    with torch.no_grad():
                        mae_loss = mae_metric(outputs, batch_y)
                        train_mae_loss.append(mae_loss.item())

                #print("i:",i)
                if (i + 1) % 100 == 0:
                    accelerator.print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                    accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                    # print(f"[{i}] allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                    # print(f"[{i}] reserved : {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
                    
                loss = loss + aux_loss
                #print("loss:",loss,args.aux_loss_factor * aux_loss)

                if args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    accelerator.backward(loss)
                    model_optim.step()

                    #probs_grad = llm_model.h[0].attn.c_proj.weight.grad
                    #print("probs_grad:", probs_grad)
                    #probs_grad = accelerator.unwrap_model(model).time_tokenizer.c_embed.weight.grad
                    # probs_grad = accelerator.unwrap_model(model).time_tokenizer.PatchAggregator.proj_v.weight.grad
                    #probs_grad = accelerator.unwrap_model(model).time_tokenizer.patch_proj.w1.weight.grad
                    #probs_grad = accelerator.unwrap_model(model).reprogramming_layer.out_projection.weight.grad
                    # print("probs_grad:", probs_grad)
                    #for name, param in model.named_parameters():
                    #    if param.grad is not None:
                    #        print(f"[USED] {name}")
                    #    else:
                    #        print(f"[NOT USED] {name}")

                if args.lradj == 'TST':
                    adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                    scheduler.step()
                torch.cuda.empty_cache()

            accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            train_mae_loss = np.average(train_mae_loss)
            print('vali:')
            vali_loss, vali_mse_loss = vali(args, accelerator, model, vali_data, vali_loader, mae_metric, mse_metric)
            print('test:')
            test_loss, test_mse_loss = vali(args, accelerator, model, test_data, test_loader, mae_metric, mse_metric)
            accelerator.print(
                "Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f} MSE Loss: {4:.7f}".format(
                    epoch + 1, train_mae_loss, vali_loss, test_loss, test_mse_loss))

            if args.lradj != 'TST':
                if args.lradj == 'COS':
                    scheduler.step()
                    accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                else:
                    if epoch == 0:
                        args.learning_rate = model_optim.param_groups[0]['lr']
                        accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                    adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)
            else:
                accelerator.print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

            early_stopping(model_optim, scheduler, vali_loss, test_loss, test_mse_loss, model, path, setting)
            if early_stopping.early_stop:
                accelerator.print("Early stopping")
                break

    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        path = './checkpoints'  # unique checkpoint saving path
        #del_files(path)  # delete checkpoint files
        #accelerator.print('success delete checkpoints')