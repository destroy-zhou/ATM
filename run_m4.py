import argparse
import torch
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from data_provider.m4 import M4Meta
from models import ATM

from data_provider.data_factory import data_provider
import time
import random
import numpy as np
import pandas

from utils.losses import smape_loss
from utils.m4_summary import M4Summary
import os

os.environ['curl_ca_bundle'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

from utils.tools import del_files, EarlyStopping, adjust_learning_rate, load_content, test

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

parser = argparse.ArgumentParser(description='ATM')
# basic config
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
parser.add_argument('--model_comment', type=str, required=False, default='none', help='prefix when saving test results')
parser.add_argument('--model', type=str, required=True, default='ATM',
                    help='model name, options: [ATM]')
parser.add_argument('--seed', type=int, default=2025, help='random seed')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
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
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# model define
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--inter_dim', type=int, default=32, help='dimension of model')
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
parser.add_argument('--use_semantic_pe', type=int, default=1, help='whether to use semantic embedding')
parser.add_argument('--use_moe', type=int, default=1, help='whether use MoE')
parser.add_argument('--router_aux_loss_factor', type=float, default=0.02, help='router auxiliary loss factor')
parser.add_argument('--apply_router_aux_loss', type=int, default=1, help='apply router auxiliary loss')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--head_dropout', type=float, default=0.1, help='head dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--model_verbose', type=int, default=0, help='whether output details')

# optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
parser.add_argument('--patience', type=int, default=4, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MAE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--percent', type=int, default=100)

args = parser.parse_args()
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

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
        setting = '{}-{}_dm{}_id{}_nh{}_pb{}_ed{}_nq{}_cl{}_ne{}_tk{}_dl{}_th{}_hf{}_af{}_dp{}_lr{}'.format(
            args.data,
            args.seasonal_patterns,
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

        if args.data == 'm4':
            args.pred_len = M4Meta.horizons_map[args.seasonal_patterns]  # Up to M4 config
            args.seq_len = 2 * args.pred_len
            args.label_len = args.pred_len
            args.frequency_map = M4Meta.frequency_map[args.seasonal_patterns]

        train_data, train_loader = data_provider(args, 'train')
        vali_data, vali_loader = data_provider(args, 'val')
        test_data, test_loader = data_provider(args, 'test')

        model_dicts = {'ATM': ATM}
        model = model_dicts[args.model].Model(args).float()

        path = os.path.join(args.checkpoints, args.task_name + '/', args.data + '/')
        if not os.path.exists(path) and accelerator.is_local_main_process:
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience, verbose=True)

        model_optim = optim.Adam(model.parameters(), lr=args.learning_rate)

        if args.lradj == 'COS':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
        else:
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=args.pct_start,
                                                epochs=args.train_epochs,
                                                max_lr=args.learning_rate)

        criterion = smape_loss()

        train_loader, vali_loader, model, model_optim, scheduler = accelerator.prepare(
            train_loader, vali_loader, model, model_optim, scheduler)

        for epoch in range(args.train_epochs):
            iter_count = 0
            train_loss = []

            model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(accelerator.device)
                batch_y = batch_y.float().to(accelerator.device)
                batch_y_mark = batch_y_mark.float().to(accelerator.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(accelerator.device)
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                    accelerator.device)

                outputs, aux_loss = model(batch_x, None, dec_inp, None)

                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:]

                batch_y_mark = batch_y_mark[:, -args.pred_len:, f_dim:]
                loss = criterion(batch_x, args.frequency_map, outputs, batch_y, batch_y_mark)

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    accelerator.print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item())
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                    accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss = loss + aux_loss

                accelerator.backward(loss)
                model_optim.step()

                if args.lradj == 'TST':
                    adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                    scheduler.step()

            accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = test(args, accelerator, model, train_loader, vali_loader, criterion)
            test_loss = vali_loss
            accelerator.print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            # early_stopping(vali_loss, model, path)  # model saving
            best_setting = early_stopping(model_optim, scheduler, vali_loss, test_loss, test_loss, model, path, setting)
            if early_stopping.early_stop:
                accelerator.print("Early stopping")
                break

            if args.lradj != 'TST':
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)
            else:
                accelerator.print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + best_setting + '/' + 'checkpoint.pt'
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        unwrapped_model.load_state_dict(torch.load(best_model_path, map_location=lambda storage, loc: storage))

        x, _ = train_loader.dataset.last_insample_window()
        y = test_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(accelerator.device)
        x = x.unsqueeze(-1)

        model.eval()

        with torch.no_grad():
            B, _, C = x.shape
            dec_inp = torch.zeros((B, args.pred_len, C)).float().to(accelerator.device)
            dec_inp = torch.cat([x[:, -args.label_len:, :], dec_inp], dim=1)
            outputs = torch.zeros((B, args.pred_len, C)).float().to(accelerator.device)
            id_list = np.arange(0, B, args.eval_batch_size)
            id_list = np.append(id_list, B)
            for i in range(len(id_list) - 1):
                outputs[id_list[i]:id_list[i + 1], :, :], aux_loss = model(
                    x[id_list[i]:id_list[i + 1]],
                    None,
                    dec_inp[id_list[i]:id_list[i + 1]],
                    None
                )
            accelerator.wait_for_everyone()
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            outputs = outputs.detach().cpu().numpy()

            preds = outputs
            trues = y
            x = x.detach().cpu().numpy()

        accelerator.print('test shape:', preds.shape)

        folder_path = './m4_results/' + args.model + '/'
        if not os.path.exists(folder_path) and accelerator.is_local_main_process:
            os.makedirs(folder_path)

        if accelerator.is_local_main_process:
            forecasts_df = pandas.DataFrame(preds[:, :, 0], columns=[f'V{i + 1}' for i in range(args.pred_len)])
            forecasts_df.index = test_loader.dataset.ids[:preds.shape[0]]
            forecasts_df.index.name = 'id'
            forecasts_df.set_index(forecasts_df.columns[0], inplace=True)
            forecasts_df.to_csv(folder_path + args.seasonal_patterns + '_forecast.csv')

            # calculate metrics
            accelerator.print(args.model)
            file_path = folder_path
            if 'Weekly_forecast.csv' in os.listdir(file_path) \
                    and 'Monthly_forecast.csv' in os.listdir(file_path) \
                    and 'Yearly_forecast.csv' in os.listdir(file_path) \
                    and 'Daily_forecast.csv' in os.listdir(file_path) \
                    and 'Hourly_forecast.csv' in os.listdir(file_path) \
                    and 'Quarterly_forecast.csv' in os.listdir(file_path):
                m4_summary = M4Summary(file_path, args.root_path)
                # m4_forecast.set_index(m4_winner_forecast.columns[0], inplace=True)
                smape_results, owa_results, mape, mase = m4_summary.evaluate()
                accelerator.print('smape:', smape_results)
                accelerator.print('mape:', mape)
                accelerator.print('mase:', mase)
                accelerator.print('owa:', owa_results)
            else:
                accelerator.print('After all 6 tasks are finished, you can calculate the averaged performance')

    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        path = './checkpoints'  # unique checkpoint saving path
        # del_files(path)  # delete checkpoint files
        # accelerator.print('success delete checkpoints')
