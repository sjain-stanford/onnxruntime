from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# ==================
import os
import logging
import random
# import h5py
from tqdm import tqdm
import datetime
import numpy as np
import dataclasses
from dataclasses import dataclass, field
from typing import Optional, Any, Dict
import json

import unittest

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, Dataset
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from fairseq import (
    checkpoint_utils,
    distributed_utils,
    options,
    quantization_utils,
    tasks,
    utils,
)
from fairseq.tasks.translation import load_langpair_dataset
from fairseq.data import Dictionary, iterators, data_utils
from fairseq.logging import meters, metrics, progress_bar

from concurrent.futures import ProcessPoolExecutor

import onnxruntime as ort
from onnxruntime.training import amp, optim, orttrainer
from onnxruntime.training.optim import PolyWarmupLRScheduler, LinearWarmupLRScheduler

# need to override torch.onnx.symbolic_opset12.nll_loss to handle ignore_index == -100 cases.
# the fix for ignore_index == -100 cases is already in pytorch master.
# however to use current torch master is causing computation changes in many tests.
# eventually we will use pytorch with fixed nll_loss once computation
# issues are understood and solved.
import onnxruntime.capi.pt_patch

# we cannot make full convergence run in nightly pipeling because of its timeout limit,
# max_steps is still needed to calculate learning rate. force_to_stop_max_steps is used to
# terminate the training before the pipeline run hit its timeout.
force_to_stop_max_steps = 2500

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def is_main_process(args):
    if hasattr(args, 'world_rank'):
        return args.world_rank in [-1, 0]
    else:
        return get_rank() == 0

def bart_model_description(args):
    batch = args.train_batch_size
    max_seq_len_in_batch = args.seq_len
    
    new_model_desc = {
        'inputs': [
            ('src_tokens', [batch, max_seq_len_in_batch],),
            ('prev_output_tokens', [batch, max_seq_len_in_batch],),
            ('target', [batch*max_seq_len_in_batch],)],
        'outputs': [
            ('loss', [], True)]}
    return new_model_desc


def create_pretraining_dataset(input_path, args):

    # train_data = pretraining_dataset(input_file=input_file, max_pred_length=max_pred_length)
    src, tgt = args.source_lang, args.target_lang
    src_dict = Dictionary.load(
            os.path.join(input_path, "dict.{}.txt".format(args.source_lang))
        )
    tgt_dict = Dictionary.load(
            os.path.join(input_path, "dict.{}.txt".format(args.target_lang))
        )
    args.padding_idx = tgt_dict.pad()
    train_data = load_langpair_dataset(
            input_path,
            'train',
            src,
            src_dict,
            tgt,
            tgt_dict,
            combine=False,
            dataset_impl=args.dataset_impl,
            upsample_primary=args.upsample_primary,
            left_pad_source=args.left_pad_source,
            left_pad_target=args.left_pad_target,
            max_source_positions=args.max_source_positions,
            max_target_positions=args.max_target_positions,
            load_alignments=args.load_alignments,
            truncate_source=args.truncate_source,
            num_buckets=args.num_batch_buckets,
            shuffle=False,
            pad_to_multiple=args.required_seq_len_multiple,
        )
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                  batch_size=args.train_batch_size * args.n_gpu, num_workers=0,
                                  pin_memory=True)
    return train_dataloader

def get_train_iterator(
    task,
    args,
    epoch,
    combine=True,
    load_dataset=True,
    data_selector=None,
    shard_batch_itr=True,
    disable_iterator_cache=False,
):
    """Return an EpochBatchIterator over the training set for a given epoch."""
    if load_dataset:
        logger.info("loading train data for epoch {}".format(epoch))
        task.load_dataset(
            args.train_subset,
            epoch=epoch,
            combine=combine,
            data_selector=data_selector,
        )
    batch_iterator = task.get_batch_iterator(
        dataset=task.dataset(args.train_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=args.seq_len,
        ignore_invalid_inputs=True,
        required_batch_size_multiple=args.required_batch_size_multiple,
        seed=args.seed,
        num_shards=args.world_size if shard_batch_itr else 1,
        shard_id=args.world_rank if shard_batch_itr else 0,
        num_workers=args.num_workers,
        epoch=epoch,
        data_buffer_size=args.data_buffer_size,
        disable_iterator_cache=disable_iterator_cache,
    )
    return batch_iterator

class pretraining_dataset(Dataset):

    def __init__(self, input_file, max_pred_length):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        f = h5py.File(input_file, "r")
        keys = ['input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions', 'masked_lm_ids',
                'next_sentence_labels']
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs[0])

    def __getitem__(self, index):
        [input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, next_sentence_labels] = [
            torch.from_numpy(input[index].astype(np.int64)) if indice < 5 else torch.from_numpy(
                np.asarray(input[index].astype(np.int64))) for indice, input in enumerate(self.inputs)]

        # HF model use default ignore_index value (-100) for CrossEntropyLoss
        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -100
        index = self.max_pred_length
        # store number of  masked tokens in index
        padded_mask_indices = (masked_lm_positions == 0).nonzero()
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]
        return [input_ids, segment_ids, input_mask,
                masked_lm_labels, next_sentence_labels]

class BARTModelWithLoss(torch.nn.Module):
    def __init__(self, model, padding_idx):
        super(BARTModelWithLoss, self).__init__()
        self.model_ = model
        self.padding_idx_ = padding_idx

    #def forward(self, src_tokens, src_lengths, prev_output_tokens, target):
    def forward(self, src_tokens, prev_output_tokens, target):
        src_lengths = None
        net_output = self.model_(src_tokens, src_lengths, prev_output_tokens, features_only=False, classification_head_name=None)
        net_out = net_output[0]

        # flatten the net_out, merging the first two dims
        net_out = net_out.view(-1, net_out.size(-1))

        #print('ORT Trainer net_out size: {}'.format(net_out.size()))
        # lprobs = self.model_.get_normalized_probs(net_out, log_probs=True)
        lprobs = F.log_softmax(net_out.float(), dim=-1)
        # lprobs = lprobs.view(-1, lprobs.size(-1))
        # target = target.view(-1)

        # loss = self.loss_fn_(net_output, lprobs, target)
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx_,
            reduction='sum',
        )
        #print('ORT Trainer loss value: {}'.format(loss))
        return loss

def add_ort_args(parser):
    group = parser.add_argument_group("ORT")

    group.add_argument("--input_dir", default=None, metavar="DIR",
                       help="The input directory where the model data is present.")
    group.add_argument("--output_dir", default=None, metavar="DIR",
                       help="The output directory where the model checkpoints will be written.")
    group.add_argument("--train_batch_size", metavar="N", default=32, type=int,
                       help="Batch size for training")
    group.add_argument("--gradient_accumulation_steps", metavar="N", default=1, type=int,
                       help="Number of updates steps to accumualte before performing a backward/update pass.")
    group.add_argument("--allreduce_post_accumulation", action="store_true",
                       help="Whether to do fp16 allreduce post accumulation.")
    group.add_argument("--local_rank", metavar="N", default=-1, type=int,
                       help="local_rank for distributed training on gpus.")
    group.add_argument("--world_rank", metavar="N", default=-1, type=int,
                       help="world_rank for distributed training on gpus.")
    group.add_argument("--world_size", metavar="N", default=1, type=int,
                       help="world_size for distributed training on gpus.")
    group.add_argument("--max_steps", metavar="N", default=1000, type=int,
                       help="Total number of training steps to perform.")
    group.add_argument("--force_num_hidden_layers", metavar="N", default=None, type=int,
                       help="Reduced number of layers.")
    group.add_argument("--padding_idx", metavar="N", default=-100, type=int,
                       help="Index of padding token.")
    group.add_argument("--seq_len", metavar="N", default=1024, type=int,
                       help="Source and target seq lengths.")
    group.add_argument("--log_freq", metavar="N", default=1, type=int,
                       help="Logging frequency.")
    group.add_argument('--warmup_proportion', default=0.01, type=float,
                       help='Proportion of training to perform linear learning rate warmup for. \
            E.g., 0.1 = 10%% of training.')
    return parser

def to_json_string(args):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(vars(args), indent=2)

def to_sanitized_dict(args) -> Dict[str, Any]:
        """
        Sanitized serialization to use with TensorBoard’s hparams
        """
        d = vars(args)
        valid_types = [bool, int, float, str, torch.Tensor]
        return {k: v if type(v) in valid_types else str(v) for k, v in d.items()}

def to_list(args) -> list:
        """
        List to use with Argparser.parse_known_args
        """
        d = args
        o=[]
        for k,v in d.items():
            o.extend(["--"+ k , str(v)])
        return o


def setup_training(args):

    assert (torch.cuda.is_available())

    if args.local_rank == -1:
        args.local_rank = 0
        args.world_rank = 0

    print("args.local_rank: ", args.local_rank)
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    args.n_gpu = 1

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if args.train_batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, batch size {} should be divisible".format(
            args.gradient_accumulation_steps, args.train_batch_size))

    # args.train_batch_size is per global step (optimization step) batch size
    # now make it a per gpu batch size
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    args.train_batch_size = args.train_batch_size // args.world_size
    args.max_sentences = args.train_batch_size

    logger.info("setup_training: args.train_batch_size = %d", args.train_batch_size)
    return device, args


def prepare_model(args, device):
    # args.encoder_embed_dim = 1024
    # args.encoder_ffn_embed_dim =1024 * 2
    args.encoder_layers = 1
    # args.encoder_attention_heads = 32
    # args.decoder_embed_dim = 1024
    # args.decoder_ffn_embed_dim = 1024 * 2
    args.decoder_layers = 1
    # args.decoder_attention_heads = 32

    task = tasks.setup_task(args)

    tgt_dict = Dictionary.load(
            os.path.join(args.input_dir, "dict.{}.txt".format(args.target_lang))
        )
    args.padding_idx = tgt_dict.pad()
    model = task.build_model(args)
    # criterion = task.build_criterion(args)
   
    # config.num_hidden_layers = 12
    # if args.force_num_hidden_layers:
    #     logger.info("Modifying model config with num_hidden_layers to %d", args.force_num_hidden_layers)
    #     config.num_hidden_layers = args.force_num_hidden_layers


    model = BARTModelWithLoss(model, args.padding_idx)
    model_desc = bart_model_description(args)

    lr_scheduler = LinearWarmupLRScheduler(total_steps=int(args.max_steps), warmup=args.warmup_proportion)

    loss_scaler = amp.DynamicLossScaler() if args.fp16 else None

    options = orttrainer.ORTTrainerOptions({'batch': {
                                                'gradient_accumulation_steps': args.gradient_accumulation_steps},
                                            'device': {'id': str(device)},
                                            'mixed_precision': {
                                                'enabled': args.fp16,
                                                'loss_scaler': loss_scaler},
                                            'debug': {'deterministic_compute': True, },
                                            'utils': {
                                                'grad_norm_clip': True},
                                            'distributed': {
                                                'world_rank': max(0, args.local_rank),
                                                'world_size': args.world_size,
                                                'local_rank': max(0, args.local_rank),
                                                'allreduce_post_accumulation': args.allreduce_post_accumulation},
                                            'lr_scheduler': lr_scheduler
                                            })

    param_optimizer = list(model.named_parameters())
    no_decay_keys = ["bias", "gamma", "beta", "LayerNorm"]
    params = [{
        'params': [n for n, p in param_optimizer if any(no_decay_key in n for no_decay_key in no_decay_keys)],
        "alpha": 0.9, "beta": 0.999, "lambda": 0.0, "epsilon": 1e-6}, {
        'params': [n for n, p in param_optimizer if not any(no_decay_key in n for no_decay_key in no_decay_keys)],
        "alpha": 0.9, "beta": 0.999, "lambda": 0.0, "epsilon": 1e-6}]

    optim_config = optim.AdamConfig(params=params, lr=args.lr[0], do_bias_correction=True)
    model = orttrainer.ORTTrainer(model, model_desc, optim_config, options=options)

    return model, task

def get_data_file(f_id, world_rank, world_size, files):
    num_files = len(files)
    if world_size > num_files:
        remainder = world_size % num_files
        return files[(f_id * world_size + world_rank + remainder * f_id) % num_files]
    elif world_size > 1:
        return files[(f_id * world_size + world_rank) % num_files]
    else:
        return files[f_id % num_files]

def pad_to_len(tokens, args):
    return data_utils.collate_tokens(tokens, args.padding_idx, pad_to_length=args.seq_len)

def main():
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)
    do_pretrain(args)


def do_pretrain(args):
    if is_main_process(args) and args.tensorboard_logdir:
        tb_writer = SummaryWriter(log_dir=args.tensorboard_logdir)
        tb_writer.add_text("args", to_json_string(args))
        tb_writer.add_hparams(to_sanitized_dict(args), metric_dict={})
    else:
        tb_writer = None

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ort.set_seed(args.seed)

    device, args = setup_training(args)

    model, task = prepare_model(args, device)

    logger.info("Running training: Batch size = %d, initial LR = %f", args.train_batch_size, args.lr[0])

    most_recent_ckpts_paths = []
    average_loss = 0.0
    epoch = 0
    training_steps = 0

    pool = ProcessPoolExecutor(1)
    while True:
        # train_dataloader = create_pretraining_dataset(
        #     args.input_dir,
        #     args)

        # train_iter = tqdm(train_dataloader, desc="Iteration") if is_main_process(args) else train_dataloader
        epoch_itr = get_train_iterator(
            task,
            args,
            1,
            # sharded data: get train iterator for next epoch
            load_dataset=True,
            # don't cache epoch iterators for sharded datasets
            disable_iterator_cache=task.has_sharded_data("train"),
            )
        itr = epoch_itr.next_epoch_itr(
            fix_batches_to_gpus=args.fix_batches_to_gpus,
            shuffle=(epoch_itr.next_epoch_idx > args.curriculum),
        )
        update_freq = (
            args.update_freq[epoch_itr.epoch - 1]
            if epoch_itr.epoch <= len(args.update_freq)
            else args.update_freq[-1]
        )
        itr = iterators.GroupedIterator(itr, update_freq)
        if getattr(args, "tpu", False):
            itr = utils.tpu_data_loader(itr)
        progress = progress_bar.progress_bar(
            itr,
            log_format=args.log_format,
            log_interval=args.log_interval,
            epoch=epoch_itr.epoch,
            tensorboard_logdir=(
                args.tensorboard_logdir if distributed_utils.is_master(args) else None
            ),
            default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
        )
        for step, batch in enumerate(progress):
            training_steps += 1
            # batch = [t.to(device) for t in batch]
            net_input = batch[0]['net_input']
            src_tokens = pad_to_len(net_input['src_tokens'], args).to(device)
            # src_lengths = net_input['src_lengths']
            # src_lengths.cpu()
            prev_output_tokens = pad_to_len(net_input['prev_output_tokens'], args).to(device)
            target = pad_to_len(batch[0]['target'], args).view(-1).to(device)
            # target = batch[0]['target'].to(device)

            loss = model.train_step(src_tokens, prev_output_tokens, target)
            average_loss += loss.item()

            global_step = model._train_step_info.optimization_step
            if training_steps % (args.log_freq * args.gradient_accumulation_steps) == 0:
                if is_main_process(args):
                    divisor = args.log_freq * args.gradient_accumulation_steps
                    if tb_writer:
                        lr = model.options.lr_scheduler.get_last_lr()[0]
                        tb_writer.add_scalar('train/summary/scalar/Learning_Rate', lr, global_step)
                        if args.fp16:
                            tb_writer.add_scalar('train/summary/scalar/loss_scale_25', loss, global_step)
                            # TODO: ORTTrainer to expose all_finite
                            # tb_writer.add_scalar('train/summary/scalar/all_fp16_gradients_finite_859', all_finite, global_step)
                        tb_writer.add_scalar('train/summary/total_loss', average_loss / divisor, global_step)
                    
                    print("Step:{} Average Loss = {}".format(global_step, average_loss / divisor))

                if global_step >= args.max_steps or global_step >= force_to_stop_max_steps:
                    if tb_writer:
                        tb_writer.close()

                    final_loss = average_loss / (args.log_freq * args.gradient_accumulation_steps)
                    return final_loss

                average_loss = 0

        del train_dataloader

        epoch += 1


def generate_tensorboard_logdir(root_dir): 
    current_date_time = datetime.datetime.today()

    dt_string = current_date_time.strftime('BERT_pretrain_%y_%m_%d_%I_%M_%S')
    return os.path.join(root_dir, dt_string)


class ORTBertPretrainTest():
    def setUp(self):
        self.output_dir = '/tmp/bert_pretrain_results'
        self.bert_model = 'bert-base-uncased'
        self.local_rank = -1
        self.world_rank = -1
        self.world_size = 1
        self.max_steps = 300000
        self.lr = 2e-5
        self.max_seq_length = 512
        self.max_predictions_per_seq = 20
        self.data = '/bert_data/megatron_bart/bin_small/'
        self.train_batch_size = 4096
        self.gradient_accumulation_steps = 64
        self.fp16 = True
        self.allreduce_post_accumulation = True
        self.tensorboard_dir = '/tmp/bert_pretrain_results'

    def test_pretrain_throughput(self):
        # setting train_batch_size and gradient_accumulation_steps to maximize per gpu memory usage under 16GB
        # to validate throughput regression.
        # train_batch_size is initially configured as per optimization batch size,
        # taking into consideration of world_size and gradient_accumulation_steps:
        # train_batch_size = world_size * gradient_accumulation_steps * batch_size_per_gpu
        # in the code later train_batch_size is translated to per gpu batch size:
        # args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps // args.world_size

        # the LAMB batch size of 64k
        optimization_batch_size = 64 * 1024
        per_gpu_batch_size = 32

        # self.train_batch_size = optimization_batch_size
        # self.gradient_accumulation_steps = optimization_batch_size // per_gpu_batch_size // self.world_size
        self.train_batch_size = 2
        self.gradient_accumulation_steps = 1

        logger.info("self.gradient_accumulation_steps = %d", self.gradient_accumulation_steps)

        # only to run on  optimization step because we only want to make sure there is no throughput regression
        self.max_steps = 10
        args = {
            "arch" : 'bart_base',
            "task" : 'translation',
            "input_dir" : self.data,
            "local_rank" : self.local_rank,
            "world_rank" : self.world_rank,
            "world_size" : self.world_size,
            "max_steps" : self.max_steps,
            "lr" : self.lr,
            "train_batch_size" : self.train_batch_size,
            "gradient_accumulation_steps" : self.gradient_accumulation_steps,
            "fp16" : self.fp16,
            "allreduce_post_accumulation" : self.allreduce_post_accumulation,
            "tensorboard-logdir" : self.tensorboard_dir,
            "output_dir" : self.output_dir,
            }
        args = to_list(args)
        parser = options.get_training_parser()
        add_ort_args(parser)
        args, extras = options.parse_args_and_arch(parser, [self.data] + args, parse_known=True)
        do_pretrain(args)

    def test_pretrain_convergence(self):
        # args = PretrainArguments(
        #     output_dir=self.output_dir,
        #     bert_model=self.bert_model,
        #     local_rank=self.local_rank,
        #     world_rank=self.world_rank,
        #     world_size=self.world_size,
        #     max_steps=self.max_steps,
        #     learning_rate=self.learning_rate,
        #     max_seq_length=self.max_seq_length,
        #     max_predictions_per_seq=self.max_predictions_per_seq,
        #     train_batch_size=self.train_batch_size,
        #     gradient_accumulation_steps=self.gradient_accumulation_steps,
        #     input_dir=self.input_dir,
        #     fp16=self.fp16,
        #     allreduce_post_accumulation=self.allreduce_post_accumulation,
        #     force_num_hidden_layers=self.force_num_hidden_layers,
        #     tensorboard_dir=generate_tensorboard_logdir('/bert_data/hf_data/test_out/'))
        # final_loss = do_pretrain(args)
        return None


# to do parallel training:
# python -m torch.distributed.launch --nproc_per_node 4 orttraining_run_bert_pretrain.py
if __name__ == "__main__":
    import sys
    logger.warning("sys.argv: %s", sys.argv)
    test = ORTBertPretrainTest()
    test.setUp()
    test.test_pretrain_throughput()
    # usage:
    #   mpirun -n 4 python orttraining_run_bert_pretrain.py ORTBertPretrainTest.test_pretrain_throughput
    #   mpirun -n 4 python orttraining_run_bert_pretrain.py ORTBertPretrainTest.test_pretrain_convergence
    #   mpirun -n 4 python orttraining_run_bert_pretrain.py     # to run real BERT convergence test
    # pytorch.distributed.launch will not work because ort backend requires MPI to broadcast ncclUniqueId
    #
    # calling unpublished get_mpi_context_xxx to get rank/size numbers.
    # from onnxruntime.capi._pybind_state import get_mpi_context_local_rank, get_mpi_context_local_size, get_mpi_context_world_rank, get_mpi_context_world_size
    # world_size = get_mpi_context_world_size()
    # if world_size > 1:
    #     print ('get_mpi_context_world_size(): ', world_size)
    #     local_rank = get_mpi_context_local_rank()

    #     if local_rank == 0:
    #         print('================================================================> os.getpid() = ', os.getpid())

    #     test = ORTBertPretrainTest()
    #     test.setUp()
    #     test.local_rank = local_rank
    #     test.world_rank = local_rank
    #     test.world_size = world_size

    #     if len(sys.argv) >= 2 and sys.argv[1] == 'ORTBertPretrainTest.test_pretrain_throughput':
    #         logger.info("running ORTBertPretrainTest.test_pretrain_throughput()...")
    #         test.test_pretrain_throughput()
    #         logger.info("ORTBertPretrainTest.test_pretrain_throughput() passed")
    #     elif len(sys.argv) >= 2 and sys.argv[1] == 'ORTBertPretrainTest.test_pretrain_convergence':
    #         logger.info("running ORTBertPretrainTest.test_pretrain_convergence()...")
    #         test.max_steps = 200
    #         test.force_num_hidden_layers = 8
    #         final_loss = test.test_pretrain_convergence()
    #         logger.info("ORTBertPretrainTest.test_pretrain_convergence() final loss = %f", final_loss)
    #         test.assertLess(final_loss, 8.5)
    #         logger.info("ORTBertPretrainTest.test_pretrain_convergence() passed")
    #     else:
    #         # https://microsoft.sharepoint.com/teams/ONNX2/_layouts/15/Doc.aspx?sourcedoc={170774be-e1c6-4f8b-a3ae-984f211fe410}&action=edit&wd=target%28ONNX%20Training.one%7C8176133b-c7cb-4ef2-aa9d-3fdad5344c40%2FGitHub%20Master%20Merge%20Schedule%7Cb67f0db1-e3a0-4add-80a6-621d67fd8107%2F%29
    #         # to make equivalent args for cpp convergence test

    #         # ngpu=4 
    #         # seq_len=128 
    #         # max_predictions_per_seq=20 
    #         # batch_size=64 
    #         # grad_acc=16 
    #         # num_train_steps=1000000 
    #         # optimizer=adam
    #         # lr=5e-4 
    #         # warmup_ratio=0.1 
    #         # warmup_mode=Linear 
    #         # effective_batch_size=$((ngpu * batch_size * grad_acc)) 
    #         # commit=$(git rev-parse HEAD | cut -c1-8) 
    #         # time_now=$(date +%m%d%H%M) 
    #         # run_name=ort_${commit}_nvbertbase_bookwiki128_fp16_${optimizer}_lr${lr}_${warmup_mode}${warmup_ratio}_g${ngpu}_bs${batch_size}_acc${grad_acc}_efbs${effective_batch_size}_steps${num_train_steps}_fp16allreduce_${time_now} 

    #         # mixed precision 
    #         # mpirun -n ${ngpu} ./onnxruntime_training_bert --model_name /bert_ort/bert_models/nv/bert-base/bert-base-uncased_L_12_H_768_A_12_V_30528_S_512_Dp_0.1_optimized_layer_norm
    #         #   --train_data_dir /bert_data/128/books_wiki_en_corpus/train --test_data_dir /bert_data/128/books_wiki_en_corpus/test
    #         #   --train_batch_size ${batch_size} --mode train --num_train_steps ${num_train_steps} --display_loss_steps 5 
    #         #   --log_dir ./logs/bert_base/${run_name} --optimizer ${optimizer} --learning_rate ${lr} --warmup_ratio ${warmup_ratio} --warmup_mode ${warmup_mode} 
    #         #   --gradient_accumulation_steps ${grad_acc} --max_predictions_per_seq=${max_predictions_per_seq} --use_mixed_precision --allreduce_in_fp16 --lambda 0
    #         #   --use_nccl | tee ${run_name}.log 

    #         test.max_seq_length = 128
    #         test.max_predictions_per_seq = 20
    #         test.gradient_accumulation_steps = 16
    #         test.train_batch_size = 64 * test.gradient_accumulation_steps * test.world_size    # cpp_batch_size (=64) * grad_acc * world_size
    #         test.max_steps = 300000

    #         test.force_num_hidden_layers = None

    #         # already using Adam (e.g. AdamConfig)
    #         test.learning_rate = 5e-4
    #         test.warmup_proportion = 0.1

    #         final_loss = test.test_pretrain_convergence()
    #         logger.info("ORTBertPretrainTest.test_pretrain_convergence() final loss = %f", final_loss)
    # else:
    #     unittest.main()
